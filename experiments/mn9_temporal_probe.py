from pathlib import Path
import csv
import json
import sys

import numpy as np


UPSTREAM = (
    Path.home()
    / "FlyQ"
    / "upstream-doomfly"
)

sys.path.insert(
    0,
    str(UPSTREAM),
)

from doom.native import NativeBrain


GRAPH = (
    UPSTREAM
    / "outputs/doom/malecns_v1/graph.npz"
)

MANIFEST = (
    UPSTREAM
    / "outputs/doom/malecns_v1/manifest.json"
)


STEP_MS = 28.6

BASELINE_STEPS = 10
RECOVERY_STEPS = 20

PULSE_LENGTHS = [
    1,
    2,
    3,
    5,
    10,
]

OUTPUT = Path(
    "telemetry/mn9_temporal_probe.csv"
)


def indexes(readouts, neuron_type):

    return {
        r["side"]: int(r["index"])
        for r in readouts
        if r.get("type") == neuron_type
    }


def spike_counts(
    counts,
    indices,
):

    left = int(
        counts[
            indices["L"]
        ]
    )

    right = int(
        counts[
            indices["R"]
        ]
    )

    return (
        left,
        right,
        left + right,
    )


def main():

    manifest = json.loads(
        MANIFEST.read_text()
    )

    readouts = manifest[
        "readouts"
    ]

    mn9 = indexes(
        readouts,
        "MN9",
    )

    dnpe = indexes(
        readouts,
        "DNpe017",
    )

    print()
    print(
        "FlyQ MN9 Temporal Probe"
    )
    print(
        "======================="
    )
    print()

    print(
        f"Neural step: {STEP_MS} ms"
    )

    print(
        f"MN9 indexes: {mn9}"
    )

    print(
        f"DNpe017 indexes: {dnpe}"
    )

    print()

    rows = []
    summaries = []

    for pulse_steps in PULSE_LENGTHS:

        print()
        print(
            "================================"
        )

        print(
            f"SUGAR PULSE: "
            f"{pulse_steps} neural step(s)"
        )

        print(
            f"Duration: "
            f"{pulse_steps * STEP_MS:.1f} ms"
        )

        print(
            "================================"
        )

        #
        # Fresh MaleCNS for each pulse length.
        #
        brain = NativeBrain(
            GRAPH
        )

        black = np.zeros(
            len(brain.uv),
            dtype=np.float32,
        )

        sequence = []

        total_steps = (
            BASELINE_STEPS
            + pulse_steps
            + RECOVERY_STEPS
        )

        for step in range(
            1,
            total_steps + 1,
        ):

            if step <= BASELINE_STEPS:

                phase = "BASELINE"
                sugar = False

            elif step <= (
                BASELINE_STEPS
                + pulse_steps
            ):

                phase = "SUGAR"
                sugar = True

            else:

                phase = "RECOVERY"
                sugar = False

            counts, wall = brain.step(
                black,
                STEP_MS,
                sugar=sugar,
            )

            (
                mn9_l,
                mn9_r,
                mn9_total,
            ) = spike_counts(
                counts,
                mn9,
            )

            (
                dnpe_l,
                dnpe_r,
                dnpe_total,
            ) = spike_counts(
                counts,
                dnpe,
            )

            relative_step = (
                step
                - BASELINE_STEPS
            )

            row = {
                "pulse_steps": (
                    pulse_steps
                ),

                "pulse_ms": (
                    pulse_steps
                    * STEP_MS
                ),

                "step": step,

                "relative_step": (
                    relative_step
                ),

                "sim_ms": (
                    step
                    * STEP_MS
                ),

                "phase": phase,

                "sugar": sugar,

                "mn9_left_spikes": (
                    mn9_l
                ),

                "mn9_right_spikes": (
                    mn9_r
                ),

                "mn9_total_spikes": (
                    mn9_total
                ),

                "dnpe017_left_spikes": (
                    dnpe_l
                ),

                "dnpe017_right_spikes": (
                    dnpe_r
                ),

                "dnpe017_total_spikes": (
                    dnpe_total
                ),

                "wall_seconds": (
                    float(wall)
                ),
            }

            rows.append(
                row
            )

            sequence.append(
                row
            )

            marker = (
                "<<<"
                if mn9_total > 0
                else ""
            )

            print(
                f"{step:>2} "
                f"| {phase:>8} "
                f"| sugar="
                f"{int(sugar)} "
                f"| MN9 "
                f"L={mn9_l:>2} "
                f"R={mn9_r:>2} "
                f"total={mn9_total:>2} "
                f"| DNpe017="
                f"{dnpe_total:>2} "
                f"{marker}"
            )

        baseline = [
            r
            for r in sequence
            if r["phase"]
            == "BASELINE"
        ]

        active = [
            r
            for r in sequence
            if r["phase"]
            == "SUGAR"
        ]

        recovery = [
            r
            for r in sequence
            if r["phase"]
            == "RECOVERY"
        ]

        baseline_mn9 = sum(
            r["mn9_total_spikes"]
            for r in baseline
        )

        sugar_mn9 = sum(
            r["mn9_total_spikes"]
            for r in active
        )

        recovery_mn9 = sum(
            r["mn9_total_spikes"]
            for r in recovery
        )

        sugar_dnpe = sum(
            r["dnpe017_total_spikes"]
            for r in active
        )

        first_active = next(
            (
                r["relative_step"]
                for r in sequence
                if (
                    r["phase"]
                    == "SUGAR"
                    and
                    r[
                        "mn9_total_spikes"
                    ]
                    > 0
                )
            ),
            None,
        )

        recovery_active_steps = [
            r["relative_step"]
            - pulse_steps
            for r in recovery
            if (
                r[
                    "mn9_total_spikes"
                ]
                > 0
            )
        ]

        last_recovery_active = (
            max(
                recovery_active_steps
            )
            if recovery_active_steps
            else 0
        )

        max_mn9_per_step = max(
            r["mn9_total_spikes"]
            for r in sequence
        )

        summaries.append({
            "pulse_steps": (
                pulse_steps
            ),

            "pulse_ms": (
                pulse_steps
                * STEP_MS
            ),

            "baseline_mn9": (
                baseline_mn9
            ),

            "sugar_mn9": (
                sugar_mn9
            ),

            "recovery_mn9": (
                recovery_mn9
            ),

            "sugar_dnpe017": (
                sugar_dnpe
            ),

            "first_active_step": (
                first_active
            ),

            "last_recovery_active": (
                last_recovery_active
            ),

            "max_mn9_per_step": (
                max_mn9_per_step
            ),
        })

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

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

        onset = (
            "NONE"
            if s[
                "first_active_step"
            ] is None
            else str(
                s[
                    "first_active_step"
                ]
            )
        )

        print(
            f"Pulse "
            f"{s['pulse_steps']:>2} steps "
            f"({s['pulse_ms']:>6.1f} ms)"
            f" | baseline MN9="
            f"{s['baseline_mn9']:>3}"
            f" | pulse MN9="
            f"{s['sugar_mn9']:>3}"
            f" | recovery MN9="
            f"{s['recovery_mn9']:>3}"
            f" | DNpe pulse="
            f"{s['sugar_dnpe017']:>3}"
            f" | onset="
            f"{onset}"
            f" | last recovery active="
            f"{s['last_recovery_active']}"
        )

    print()
    print(
        "Interpretation"
    )

    print(
        "--------------"
    )

    print(
        "Ideal independent event channel:"
    )

    print(
        "  baseline MN9 = 0"
    )

    print(
        "  rapid MN9 onset during sugar"
    )

    print(
        "  low DNpe017 during sugar"
    )

    print(
        "  short or zero MN9 persistence"
    )

    print()

    print(
        "If one 28.6 ms sugar step is"
    )

    print(
        "sufficient and recovery is clean,"
    )

    print(
        "the next milestone is an MN9"
    )

    print(
        "event/debounce attack decoder."
    )

    print()

    print(
        f"Telemetry saved → "
        f"{OUTPUT}"
    )


if __name__ == "__main__":
    main()
