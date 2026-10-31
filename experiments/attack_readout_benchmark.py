from pathlib import Path
import csv
import json
import sys

import numpy as np


# ============================================================
# Upstream DoomFly
# ============================================================

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
from doom.engine import NeuralControls
from doom.game import retinal_samples


GRAPH = (
    UPSTREAM
    / "outputs/doom/malecns_v1/graph.npz"
)

MANIFEST = (
    UPSTREAM
    / "outputs/doom/malecns_v1/manifest.json"
)


# ============================================================
# Experiment
# ============================================================

WIDTH = 320
HEIGHT = 240

DURATION_MS = 500.0

OUTPUT = Path(
    "telemetry/attack_readout_benchmark.csv"
)


# ============================================================
# RGB stimuli
# ============================================================

def make_frame(condition: str) -> np.ndarray:

    frame = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )

    if condition == "black":

        pass

    elif condition == "white":

        frame[:] = 255

    elif condition == "left_half":

        frame[
            :,
            : WIDTH // 2,
        ] = 255

    elif condition == "right_half":

        frame[
            :,
            WIDTH // 2 :,
        ] = 255

    elif condition == "center_band":

        start = WIDTH // 3
        end = 2 * WIDTH // 3

        frame[
            :,
            start:end,
        ] = 255

    elif condition == "target_center":

        #
        # Same approximate target geometry
        # used in FlyQ navigation.
        #
        target_width = 64
        target_height = 160

        x0 = (
            WIDTH // 2
            - target_width // 2
        )

        x1 = (
            x0
            + target_width
        )

        y0 = (
            HEIGHT // 2
            - target_height // 2
        )

        y1 = (
            y0
            + target_height
        )

        frame[
            y0:y1,
            x0:x1,
        ] = 255

    elif condition == "movement_outer":

        #
        # Same style of bilateral outer
        # movement cue used in FlyQ's
        # DNpe017 movement experiments.
        #
        band = int(
            WIDTH * 0.30
        )

        frame[
            :,
            :band,
        ] = 188

        frame[
            :,
            WIDTH - band :,
        ] = 188

    else:

        raise ValueError(
            condition
        )

    return frame


# ============================================================
# Readout helpers
# ============================================================

def readouts_of_type(
    readouts,
    neuron_type,
):

    return [
        r
        for r in readouts
        if r.get("type")
        == neuron_type
    ]


def raw_spikes(
    readouts,
    counts,
    neuron_type,
):

    selected = readouts_of_type(
        readouts,
        neuron_type,
    )

    return sum(
        int(
            counts[
                int(r["index"])
            ]
        )
        for r in selected
    )


def raw_rate_hz(
    readouts,
    counts,
    neuron_type,
    duration_seconds,
):

    return (
        raw_spikes(
            readouts,
            counts,
            neuron_type,
        )
        / duration_seconds
    )


# ============================================================
# Single controlled condition
# ============================================================

def run_condition(
    condition_name,
    frame_name,
    sugar,
    readouts,
):

    #
    # Fresh MaleCNS for every controlled
    # condition.
    #
    # This is deliberate:
    # we want stimulus-response
    # characterization without previous
    # neural-state carryover.
    #
    brain = NativeBrain(
        GRAPH
    )

    frame = make_frame(
        frame_name
    )

    light = retinal_samples(
        frame,
        brain.uv,
    )

    counts, wall = brain.step(
        light,
        DURATION_MS,
        sugar=sugar,
    )

    seconds = (
        DURATION_MS
        / 1000.0
    )

    mn9_spikes = raw_spikes(
        readouts,
        counts,
        "MN9",
    )

    mn9_hz = raw_rate_hz(
        readouts,
        counts,
        "MN9",
        seconds,
    )

    dnpe_spikes = raw_spikes(
        readouts,
        counts,
        "DNpe017",
    )

    dnpe_hz = raw_rate_hz(
        readouts,
        counts,
        "DNpe017",
        seconds,
    )

    dnp20_spikes = raw_spikes(
        readouts,
        counts,
        "DNp20",
    )

    #
    # Decode the SAME raw neural output
    # using DoomFly's two fixed decoder
    # definitions.
    #
    biological = NeuralControls(
        readouts,
        mode="biological",
    )

    bci = NeuralControls(
        readouts,
        mode="bci",
    )

    bio_action = biological.decode(
        counts,
        seconds,
    )

    bci_action = bci.decode(
        counts,
        seconds,
    )

    return {
        "condition": condition_name,
        "frame": frame_name,
        "sugar": bool(sugar),

        "retina_mean": float(
            light.mean()
        ),

        "retina_active": int(
            np.count_nonzero(
                light
            )
        ),

        "mn9_spikes": int(
            mn9_spikes
        ),

        "mn9_raw_hz": float(
            mn9_hz
        ),

        "dnpe017_spikes": int(
            dnpe_spikes
        ),

        "dnpe017_raw_hz": float(
            dnpe_hz
        ),

        "dnp20_spikes": int(
            dnp20_spikes
        ),

        #
        # DoomFly biological decoder:
        # attack = MN9 spike.
        #
        "bio_attack": bool(
            bio_action["attack"]
        ),

        #
        # DoomFly BCI decoder:
        # attack = DNpe017 spike.
        #
        "bci_attack": bool(
            bci_action["attack"]
        ),

        "bci_forward": float(
            bci_action["forward"]
        ),

        "wall_seconds": float(
            wall
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    manifest = json.loads(
        MANIFEST.read_text()
    )

    readouts = manifest[
        "readouts"
    ]

    mn9_readouts = (
        readouts_of_type(
            readouts,
            "MN9",
        )
    )

    dnpe_readouts = (
        readouts_of_type(
            readouts,
            "DNpe017",
        )
    )

    print()
    print(
        "FlyQ MN9 Attack Readout Benchmark"
    )
    print(
        "================================="
    )
    print()

    print(
        "Question:"
    )

    print(
        "Can MN9 provide an independent"
    )

    print(
        "neural event channel distinct"
    )

    print(
        "from DNpe017 locomotion?"
    )

    print()

    print(
        f"MN9 readouts found: "
        f"{len(mn9_readouts)}"
    )

    for r in mn9_readouts:

        print(
            "  ",
            {
                k: r.get(k)
                for k in (
                    "type",
                    "side",
                    "index",
                    "id",
                    "root_id",
                )
                if k in r
            },
        )

    print()

    print(
        f"DNpe017 readouts found: "
        f"{len(dnpe_readouts)}"
    )

    for r in dnpe_readouts:

        print(
            "  ",
            {
                k: r.get(k)
                for k in (
                    "type",
                    "side",
                    "index",
                    "id",
                    "root_id",
                )
                if k in r
            },
        )

    print()

    # ========================================================
    # Controlled conditions
    # ========================================================

    conditions = [

        #
        # Baseline
        #
        (
            "black",
            "black",
            False,
        ),

        #
        # Visual-only conditions
        #
        (
            "white",
            "white",
            False,
        ),

        (
            "left_half",
            "left_half",
            False,
        ),

        (
            "center_band",
            "center_band",
            False,
        ),

        (
            "right_half",
            "right_half",
            False,
        ),

        (
            "target_center",
            "target_center",
            False,
        ),

        #
        # Strong known DNpe017 visual cue.
        #
        (
            "movement_outer",
            "movement_outer",
            False,
        ),

        #
        # Existing DoomFly sugar sensory
        # drive. This is NOT an attack
        # command.
        #
        (
            "black_plus_sugar",
            "black",
            True,
        ),

        (
            "target_plus_sugar",
            "target_center",
            True,
        ),

        (
            "movement_plus_sugar",
            "movement_outer",
            True,
        ),
    ]

    results = []

    for (
        condition_name,
        frame_name,
        sugar,
    ) in conditions:

        print(
            f"Running "
            f"{condition_name} ...",
            flush=True,
        )

        result = run_condition(
            condition_name,
            frame_name,
            sugar,
            readouts,
        )

        results.append(
            result
        )

        print(
            f"{condition_name:>20} | "
            f"retina="
            f"{result['retina_mean']:.4f} | "
            f"MN9="
            f"{result['mn9_spikes']:>4} spikes "
            f"({result['mn9_raw_hz']:>7.2f} Hz) | "
            f"DNpe017="
            f"{result['dnpe017_spikes']:>4} spikes "
            f"({result['dnpe017_raw_hz']:>7.2f} Hz) | "
            f"bio_attack="
            f"{str(result['bio_attack']):>5} | "
            f"bci_attack="
            f"{str(result['bci_attack']):>5}"
        )

    # ========================================================
    # Save
    # ========================================================

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
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    # ========================================================
    # Analysis
    # ========================================================

    print()
    print(
        "================================="
    )
    print(
        "SUMMARY"
    )
    print(
        "================================="
    )
    print()

    visual_results = [
        r
        for r in results
        if not r["sugar"]
    ]

    sugar_results = [
        r
        for r in results
        if r["sugar"]
    ]

    visual_mn9_active = [
        r
        for r in visual_results
        if r["mn9_spikes"] > 0
    ]

    sugar_mn9_active = [
        r
        for r in sugar_results
        if r["mn9_spikes"] > 0
    ]

    print(
        "MN9 visual-active conditions:"
    )

    if visual_mn9_active:

        for r in visual_mn9_active:

            print(
                f"  {r['condition']}: "
                f"{r['mn9_spikes']} spikes "
                f"({r['mn9_raw_hz']:.2f} Hz)"
            )

    else:

        print(
            "  NONE"
        )

    print()

    print(
        "MN9 sugar-active conditions:"
    )

    if sugar_mn9_active:

        for r in sugar_mn9_active:

            print(
                f"  {r['condition']}: "
                f"{r['mn9_spikes']} spikes "
                f"({r['mn9_raw_hz']:.2f} Hz)"
            )

    else:

        print(
            "  NONE"
        )

    print()

    #
    # Find conditions with MN9 activity
    # and comparatively low DNpe017.
    #
    mn9_active = [
        r
        for r in results
        if r["mn9_spikes"] > 0
    ]

    selective = sorted(
        mn9_active,
        key=lambda r: (
            r["dnpe017_raw_hz"],
            -r["mn9_raw_hz"],
        ),
    )

    print(
        "Best MN9 / low-DNpe017 conditions:"
    )

    if selective:

        for r in selective[:5]:

            print(
                f"  {r['condition']:>20}: "
                f"MN9="
                f"{r['mn9_raw_hz']:7.2f} Hz | "
                f"DNpe017="
                f"{r['dnpe017_raw_hz']:7.2f} Hz"
            )

    else:

        print(
            "  No MN9 spikes observed."
        )

    print()
    print(
        "Interpretation guide"
    )
    print(
        "--------------------"
    )

    print(
        "1. MN9 silent everywhere:"
    )
    print(
        "   Reject MN9 for the current FlyQ"
    )
    print(
        "   sensory/control architecture."
    )

    print()

    print(
        "2. MN9 responds to visual input:"
    )
    print(
        "   Candidate independent event"
    )
    print(
        "   channel; characterize temporal"
    )
    print(
        "   selectivity next."
    )

    print()

    print(
        "3. MN9 responds only with sugar:"
    )
    print(
        "   Independent neural activity exists,"
    )
    print(
        "   but it is gustatory-drive dependent."
    )
    print(
        "   Do NOT call this natural attack."
    )

    print()

    print(
        "4. MN9 always follows DNpe017:"
    )
    print(
        "   Poor independent action channel;"
    )
    print(
        "   look for another readout."
    )

    print()

    print(
        f"Telemetry saved → "
        f"{OUTPUT}"
    )


if __name__ == "__main__":
    main()
