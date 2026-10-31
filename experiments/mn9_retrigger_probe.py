from pathlib import Path
import json
import sys

import numpy as np

UPSTREAM = Path.home() / "FlyQ" / "upstream-doomfly"
sys.path.insert(0, str(UPSTREAM))

from doom.native import NativeBrain

GRAPH = UPSTREAM / "outputs/doom/malecns_v1/graph.npz"
MANIFEST = UPSTREAM / "outputs/doom/malecns_v1/manifest.json"

STEP_MS = 28.6
BASELINE_STEPS = 10

PULSE_LENGTHS = [2, 10]

MAX_RECOVERY_STEPS = 200
ZERO_CONFIRM_STEPS = 5
SECOND_RECOVERY_STEPS = 40


def indexes(readouts, neuron_type):
    return {
        r["side"]: int(r["index"])
        for r in readouts
        if r.get("type") == neuron_type
    }


def spikes(counts, indices):
    left = int(counts[indices["L"]])
    right = int(counts[indices["R"]])
    return left, right, left + right


class MN9EventDecoder:
    """
    One MN9 activation episode -> ONE attack event.

    Persistent MN9 activity is ignored until
    MN9 has been silent for several consecutive
    neural windows.
    """

    def __init__(self, zero_confirm_steps=5):
        self.zero_confirm_steps = zero_confirm_steps
        self.armed = True
        self.zero_run = 0
        self.events = 0

    def update(self, spike_count):
        attack_event = False

        if spike_count > 0:
            self.zero_run = 0

            if self.armed:
                attack_event = True
                self.events += 1
                self.armed = False

        else:
            self.zero_run += 1

            if (
                not self.armed
                and self.zero_run >= self.zero_confirm_steps
            ):
                self.armed = True

        return attack_event


def step_brain(brain, light, sugar, mn9, dnpe):
    counts, _ = brain.step(
        light,
        STEP_MS,
        sugar=sugar,
    )

    mn9_l, mn9_r, mn9_total = spikes(
        counts,
        mn9,
    )

    dnpe_l, dnpe_r, dnpe_total = spikes(
        counts,
        dnpe,
    )

    return {
        "mn9_l": mn9_l,
        "mn9_r": mn9_r,
        "mn9": mn9_total,
        "dnpe_l": dnpe_l,
        "dnpe_r": dnpe_r,
        "dnpe": dnpe_total,
    }


def main():
    manifest = json.loads(
        MANIFEST.read_text()
    )

    readouts = manifest["readouts"]

    mn9 = indexes(
        readouts,
        "MN9",
    )

    dnpe = indexes(
        readouts,
        "DNpe017",
    )

    print()
    print("FlyQ MN9 Retrigger Probe")
    print("========================")
    print()

    print(f"Neural step: {STEP_MS} ms")
    print(
        f"Re-arm rule: "
        f"{ZERO_CONFIRM_STEPS} consecutive MN9-zero steps"
    )
    print()

    summaries = []

    for pulse_steps in PULSE_LENGTHS:
        print()
        print("================================")
        print(
            f"PULSE LENGTH = {pulse_steps} steps "
            f"({pulse_steps * STEP_MS:.1f} ms)"
        )
        print("================================")

        # ONE persistent MaleCNS for both pulses.
        brain = NativeBrain(GRAPH)

        light = np.zeros(
            len(brain.uv),
            dtype=np.float32,
        )

        decoder = MN9EventDecoder(
            ZERO_CONFIRM_STEPS
        )

        sim_step = 0

        first_event_step = None
        second_event_step = None

        # Baseline
        for _ in range(BASELINE_STEPS):
            sim_step += 1

            result = step_brain(
                brain,
                light,
                False,
                mn9,
                dnpe,
            )

            event = decoder.update(
                result["mn9"]
            )

            print(
                f"{sim_step:>3} | BASELINE "
                f"| MN9={result['mn9']:>2} "
                f"| DNpe={result['dnpe']:>2} "
                f"| armed={decoder.armed} "
                f"| event={event}"
            )

        # First pulse
        print()
        print(">>> FIRST SUGAR PULSE")

        for pulse_step in range(
            1,
            pulse_steps + 1,
        ):
            sim_step += 1

            result = step_brain(
                brain,
                light,
                True,
                mn9,
                dnpe,
            )

            event = decoder.update(
                result["mn9"]
            )

            if event and first_event_step is None:
                first_event_step = sim_step

            print(
                f"{sim_step:>3} | SUGAR-1 "
                f"| pulse={pulse_step:>2} "
                f"| MN9={result['mn9']:>2} "
                f"| DNpe={result['dnpe']:>2} "
                f"| armed={decoder.armed} "
                f"| ATTACK_EVENT={event}"
            )

        # Recovery / wait for re-arm
        print()
        print(">>> RECOVERY / WAIT FOR RE-ARM")

        rearm_recovery_step = None
        previous_armed = decoder.armed

        for recovery_step in range(
            1,
            MAX_RECOVERY_STEPS + 1,
        ):
            sim_step += 1

            result = step_brain(
                brain,
                light,
                False,
                mn9,
                dnpe,
            )

            event = decoder.update(
                result["mn9"]
            )

            if (
                not previous_armed
                and decoder.armed
            ):
                rearm_recovery_step = recovery_step

                print(
                    f"{sim_step:>3} | RECOVERY "
                    f"| r={recovery_step:>3} "
                    f"| MN9={result['mn9']:>2} "
                    f"| DNpe={result['dnpe']:>2} "
                    f"| RE-ARMED ✅"
                )

                break

            if (
                recovery_step <= 10
                or recovery_step % 10 == 0
                or result["mn9"] == 0
            ):
                print(
                    f"{sim_step:>3} | RECOVERY "
                    f"| r={recovery_step:>3} "
                    f"| MN9={result['mn9']:>2} "
                    f"| DNpe={result['dnpe']:>2} "
                    f"| armed={decoder.armed}"
                )

            previous_armed = decoder.armed

        if rearm_recovery_step is None:
            print()
            print(
                "NO RE-ARM within "
                f"{MAX_RECOVERY_STEPS} steps "
                f"({MAX_RECOVERY_STEPS * STEP_MS:.1f} ms) ❌"
            )

            summaries.append({
                "pulse_steps": pulse_steps,
                "pulse_ms": pulse_steps * STEP_MS,
                "first_event": first_event_step,
                "rearm_steps": None,
                "rearm_ms": None,
                "second_event": None,
                "event_count": decoder.events,
            })

            continue

        # Second identical pulse, same brain
        print()
        print(">>> SECOND SUGAR PULSE")

        for pulse_step in range(
            1,
            pulse_steps + 1,
        ):
            sim_step += 1

            result = step_brain(
                brain,
                light,
                True,
                mn9,
                dnpe,
            )

            event = decoder.update(
                result["mn9"]
            )

            if event and second_event_step is None:
                second_event_step = sim_step

            print(
                f"{sim_step:>3} | SUGAR-2 "
                f"| pulse={pulse_step:>2} "
                f"| MN9={result['mn9']:>2} "
                f"| DNpe={result['dnpe']:>2} "
                f"| armed={decoder.armed} "
                f"| ATTACK_EVENT={event}"
            )

        # Second recovery
        print()
        print(">>> SECOND RECOVERY")

        for recovery_step in range(
            1,
            SECOND_RECOVERY_STEPS + 1,
        ):
            sim_step += 1

            result = step_brain(
                brain,
                light,
                False,
                mn9,
                dnpe,
            )

            event = decoder.update(
                result["mn9"]
            )

            if (
                recovery_step <= 5
                or recovery_step % 10 == 0
            ):
                print(
                    f"{sim_step:>3} | RECOVERY-2 "
                    f"| r={recovery_step:>2} "
                    f"| MN9={result['mn9']:>2} "
                    f"| DNpe={result['dnpe']:>2} "
                    f"| event={event}"
                )

        summaries.append({
            "pulse_steps": pulse_steps,
            "pulse_ms": pulse_steps * STEP_MS,
            "first_event": first_event_step,
            "rearm_steps": rearm_recovery_step,
            "rearm_ms": rearm_recovery_step * STEP_MS,
            "second_event": second_event_step,
            "event_count": decoder.events,
        })

    print()
    print("================================")
    print("FINAL SUMMARY")
    print("================================")
    print()

    for s in summaries:
        print(
            f"Pulse {s['pulse_steps']:>2} "
            f"({s['pulse_ms']:>6.1f} ms)"
            f" | first_event={s['first_event']}"
            f" | rearm_steps={s['rearm_steps']}"
            f" | rearm_ms={s['rearm_ms']}"
            f" | second_event={s['second_event']}"
            f" | total_events={s['event_count']}"
        )

    print()
    print("Desired result:")
    print("  one event per sugar episode")
    print("  no repeated events during persistent MN9 firing")
    print("  eventual zero-state re-arm")
    print("  second pulse produces a second event")


if __name__ == "__main__":
    main()
