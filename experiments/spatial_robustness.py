from pathlib import Path
import gc
import json
import sys

import numpy as np


UPSTREAM = Path.home() / "FlyQ" / "upstream-doomfly"
sys.path.insert(0, str(UPSTREAM))

from doom.native import NativeBrain
from doom.engine import NeuralControls


GRAPH = UPSTREAM / "outputs/doom/malecns_v1/graph.npz"
MANIFEST = UPSTREAM / "outputs/doom/malecns_v1/manifest.json"

STIMULATION_MS = 500
INTENSITY = 0.5

COUNTS = [100, 400, 800]
REPEATS = 3

RNG = np.random.default_rng(42)


def find_readout(readouts, neuron_type, side):
    for r in readouts:
        if (
            r["type"] == neuron_type
            and r["side"] == side
        ):
            return r

    raise RuntimeError(
        f"Missing {neuron_type} {side}"
    )


def make_regions(brain):
    x = brain.uv[:, 0]

    order = np.argsort(x)

    third = len(order) // 3

    return {
        "left": order[:third],
        "center": order[third:2 * third],
        "right": order[-third:],
    }


def measure(light, manifest):
    brain = NativeBrain(GRAPH)

    controls = NeuralControls(
        manifest["readouts"],
        mode="bci",
    )

    counts, wall = brain.step(
        light,
        STIMULATION_MS,
    )

    action = controls.decode(
        counts,
        STIMULATION_MS / 1000,
    )

    readouts = action["readouts"]

    left = find_readout(
        readouts,
        "DNp20",
        "L",
    )

    right = find_readout(
        readouts,
        "DNp20",
        "R",
    )

    result = {
        "left_hz": float(left["rate_hz"]),
        "right_hz": float(right["rate_hz"]),
        "turn": float(
            right["rate_hz"]
            - left["rate_hz"]
        ),
        "wall": float(wall),
    }

    del brain
    gc.collect()

    return result


def get_template_brain():
    brain = NativeBrain(GRAPH)

    regions = make_regions(brain)
    receptor_count = len(brain.uv)

    del brain
    gc.collect()

    return receptor_count, regions


def main():
    manifest = json.loads(
        MANIFEST.read_text()
    )

    receptor_count, regions = get_template_brain()

    print()
    print("FlyQ Spatial Robustness Probe")
    print("=============================")
    print(
        f"Retinal inputs : {receptor_count}"
    )
    print(
        f"Intensity      : {INTENSITY}"
    )
    print(
        f"Stimulation    : {STIMULATION_MS} ms"
    )
    print()

    # Baseline
    black = np.zeros(
        receptor_count,
        dtype=np.float32,
    )

    baseline = measure(
        black,
        manifest,
    )

    baseline_turn = baseline["turn"]

    print(
        f"BLACK baseline turn = "
        f"{baseline_turn:.2f} Hz"
    )
    print()

    all_results = {}

    for active_count in COUNTS:

        print(
            f"=== {active_count} receptors ==="
        )

        for condition in [
            "left",
            "center",
            "right",
        ]:

            deltas = []

            for repeat in range(REPEATS):

                available = regions[condition]

                chosen = RNG.choice(
                    available,
                    size=active_count,
                    replace=False,
                )

                light = np.zeros(
                    receptor_count,
                    dtype=np.float32,
                )

                light[chosen] = INTENSITY

                result = measure(
                    light,
                    manifest,
                )

                delta = (
                    result["turn"]
                    - baseline_turn
                )

                deltas.append(delta)

                print(
                    f"{condition:>6} "
                    f"N={active_count:>4} "
                    f"trial={repeat + 1} | "
                    f"turn={result['turn']:7.2f} "
                    f"Δ={delta:+7.2f}"
                )

            mean = float(
                np.mean(deltas)
            )

            std = float(
                np.std(deltas)
            )

            all_results[
                (active_count, condition)
            ] = (mean, std)

        print()

    print()
    print("SUMMARY")
    print("=======")
    print(
        "Values are Δturn from black "
        "(mean ± SD)"
    )
    print()

    for active_count in COUNTS:

        print(
            f"N = {active_count}"
        )

        for condition in [
            "left",
            "center",
            "right",
        ]:

            mean, std = all_results[
                (active_count, condition)
            ]

            print(
                f"  {condition:>6}: "
                f"{mean:+7.2f} ± "
                f"{std:5.2f} Hz"
            )

        print()


if __name__ == "__main__":
    main()
