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

# Frozen candidate attack stimulus
PULSE_STEPS = 2
PULSE_MS = PULSE_STEPS * STEP_MS

N_ATTACKS = 5

# Require sustained silence.
ZERO_CONFIRM_STEPS = 5

# Prevent us from waiting forever.
MAX_RECOVERY_STEPS = 300


def indexes(readouts, neuron_type):
    return {
        r["side"]: int(r["index"])
        for r in readouts
        if r.get("type") == neuron_type
    }


def total_spikes(counts, indices):
    return (
        int(counts[indices["L"]])
        + int(counts[indices["R"]])
    )


class MN9EventDecoder:
    def __init__(self, zero_confirm_steps=5):
        self.zero_confirm_steps = zero_confirm_steps

        self.armed = True
        self.zero_run = 0
        self.events = 0

    def update(self, mn9_spikes):
        event = False

        if mn9_spikes > 0:
            self.zero_run = 0

            if self.armed:
                event = True
                self.events += 1
                self.armed = False

        else:
            self.zero_run += 1

            if (
                not self.armed
                and self.zero_run
                >= self.zero_confirm_steps
            ):
                self.armed = True

        return event


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

    brain = NativeBrain(
        GRAPH
    )

    black = np.zeros(
        len(brain.uv),
        dtype=np.float32,
    )

    decoder = MN9EventDecoder(
        ZERO_CONFIRM_STEPS
    )

    print()
    print("FlyQ Repeated MN9 Attack Probe")
    print("==============================")
    print()

    print(
        f"Attack stimulus: "
        f"{PULSE_STEPS} sugar steps "
        f"({PULSE_MS:.1f} ms)"
    )

    print(
        f"Requested attacks: {N_ATTACKS}"
    )

    print(
        "ONE persistent MaleCNS brain."
    )

    print(
        "Neural resets: 0"
    )

    print()

    sim_step = 0

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    for _ in range(BASELINE_STEPS):

        sim_step += 1

        counts, _ = brain.step(
            black,
            STEP_MS,
            sugar=False,
        )

        mn9_count = total_spikes(
            counts,
            mn9,
        )

        dnpe_count = total_spikes(
            counts,
            dnpe,
        )

        event = decoder.update(
            mn9_count
        )

        print(
            f"{sim_step:>4} "
            f"| BASELINE "
            f"| MN9={mn9_count:>2} "
            f"| DNpe={dnpe_count:>2} "
            f"| event={event}"
        )

    summaries = []

    # --------------------------------------------------------
    # Repeated attack episodes
    # --------------------------------------------------------

    for attack_number in range(
        1,
        N_ATTACKS + 1,
    ):

        print()
        print(
            "================================"
        )

        print(
            f"ATTACK REQUEST "
            f"{attack_number}/{N_ATTACKS}"
        )

        print(
            "================================"
        )

        event_this_episode = 0

        # ----------------------------------------------------
        # Sugar pulse
        # ----------------------------------------------------

        for pulse_step in range(
            1,
            PULSE_STEPS + 1,
        ):

            sim_step += 1

            counts, _ = brain.step(
                black,
                STEP_MS,
                sugar=True,
            )

            mn9_count = total_spikes(
                counts,
                mn9,
            )

            dnpe_count = total_spikes(
                counts,
                dnpe,
            )

            event = decoder.update(
                mn9_count
            )

            if event:
                event_this_episode += 1

            print(
                f"{sim_step:>4} "
                f"| SUGAR "
                f"| pulse={pulse_step} "
                f"| MN9={mn9_count:>2} "
                f"| DNpe={dnpe_count:>2} "
                f"| armed={decoder.armed} "
                f"| ATTACK_EVENT={event}"
            )

        # ----------------------------------------------------
        # Recovery
        # ----------------------------------------------------

        recovery_steps = 0

        while (
            recovery_steps
            < MAX_RECOVERY_STEPS
        ):

            recovery_steps += 1
            sim_step += 1

            counts, _ = brain.step(
                black,
                STEP_MS,
                sugar=False,
            )

            mn9_count = total_spikes(
                counts,
                mn9,
            )

            dnpe_count = total_spikes(
                counts,
                dnpe,
            )

            was_armed = (
                decoder.armed
            )

            event = decoder.update(
                mn9_count
            )

            if event:
                event_this_episode += 1

            if (
                recovery_steps <= 10
                or
                recovery_steps % 20 == 0
                or
                decoder.armed
                != was_armed
            ):

                print(
                    f"{sim_step:>4} "
                    f"| RECOVERY "
                    f"| r={recovery_steps:>3} "
                    f"| MN9={mn9_count:>2} "
                    f"| DNpe={dnpe_count:>2} "
                    f"| armed={decoder.armed} "
                    f"| event={event}"
                )

            if decoder.armed:

                print(
                    f">>> RE-ARMED after "
                    f"{recovery_steps} steps "
                    f"("
                    f"{recovery_steps * STEP_MS:.1f} ms"
                    f")"
                )

                break

        rearmed = decoder.armed

        if not rearmed:

            print(
                f">>> FAILED TO RE-ARM "
                f"within "
                f"{MAX_RECOVERY_STEPS} steps ❌"
            )

        summaries.append({
            "attack": attack_number,
            "events": event_this_episode,
            "recovery_steps": (
                recovery_steps
            ),
            "recovery_ms": (
                recovery_steps
                * STEP_MS
            ),
            "rearmed": rearmed,
        })

        if not rearmed:
            break

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        "================================"
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "================================"
    )

    print()

    for s in summaries:

        status = (
            "✅"
            if (
                s["events"] == 1
                and s["rearmed"]
            )
            else "❌"
        )

        print(
            f"Attack "
            f"{s['attack']} "
            f"| events="
            f"{s['events']} "
            f"| recovery="
            f"{s['recovery_steps']} steps "
            f"("
            f"{s['recovery_ms']:.1f} ms"
            f") "
            f"| rearmed="
            f"{s['rearmed']} "
            f"| {status}"
        )

    print()

    print(
        f"Total decoder events: "
        f"{decoder.events}"
    )

    print(
        f"Requested attack episodes: "
        f"{N_ATTACKS}"
    )

    print()

    passed = (
        len(summaries) == N_ATTACKS
        and
        decoder.events == N_ATTACKS
        and
        all(
            s["events"] == 1
            and
            s["rearmed"]
            for s in summaries
        )
    )

    if passed:

        print(
            "REPEATED ATTACK CHANNEL: PASS ✅"
        )

    else:

        print(
            "REPEATED ATTACK CHANNEL: FAIL ❌"
        )


if __name__ == "__main__":
    main()
