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

ACTIVE_RECEPTORS = 800
INTENSITY = 0.5

BASELINE_MS = 100
WINDOW_MS = 25
TOTAL_STIMULUS_MS = 500


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


def make_stimulus(brain, condition):
    uv = brain.uv
    x = uv[:, 0]

    light = np.zeros(
        len(x),
        dtype=np.float32,
    )

    if condition == "left":
        indices = np.argsort(x)[:ACTIVE_RECEPTORS]

    elif condition == "right":
        indices = np.argsort(x)[-ACTIVE_RECEPTORS:]

    elif condition == "center":
        indices = np.argsort(
            np.abs(x - 0.5)
        )[:ACTIVE_RECEPTORS]

    else:
        raise ValueError(condition)

    light[indices] = INTENSITY

    return light


def decode_window(
    controls,
    counts,
    duration_ms,
):
    action = controls.decode(
        counts,
        duration_ms / 1000,
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

    return {
        "left": float(left["rate_hz"]),
        "right": float(right["rate_hz"]),
        "turn": float(
            right["rate_hz"]
            - left["rate_hz"]
        ),
    }


def run_trial(condition, manifest):
    brain = NativeBrain(GRAPH)

    controls = NeuralControls(
        manifest["readouts"],
        mode="bci",
    )

    dark = np.zeros(
        len(brain.uv),
        dtype=np.float32,
    )

    stimulus = make_stimulus(
        brain,
        condition,
    )

    # Initial dark period.
    baseline_counts, _ = brain.step(
        dark,
        BASELINE_MS,
    )

    baseline = decode_window(
        controls,
        baseline_counts,
        BASELINE_MS,
    )

    results = []

    elapsed = 0

    while elapsed < TOTAL_STIMULUS_MS:

        counts, wall = brain.step(
            stimulus,
            WINDOW_MS,
        )

        elapsed += WINDOW_MS

        response = decode_window(
            controls,
            counts,
            WINDOW_MS,
        )

        results.append({
            "time_ms": elapsed,
            "turn": response["turn"],
            "delta": (
                response["turn"]
                - baseline["turn"]
            ),
            "left": response["left"],
            "right": response["right"],
            "wall": wall,
        })

    del brain
    gc.collect()

    return baseline, results


def main():
    manifest = json.loads(
        MANIFEST.read_text()
    )

    print()
    print("FlyQ Temporal Response Probe")
    print("============================")
    print(
        f"Receptors : {ACTIVE_RECEPTORS}"
    )
    print(
        f"Intensity : {INTENSITY}"
    )
    print(
        f"Window    : {WINDOW_MS} ms"
    )
    print()

    for condition in [
        "left",
        "center",
        "right",
    ]:

        print()
        print(
            f"=== {condition.upper()} ==="
        )

        baseline, results = run_trial(
            condition,
            manifest,
        )

        print(
            f"baseline turn = "
            f"{baseline['turn']:.2f} Hz"
        )

        for r in results:
            print(
                f"{r['time_ms']:>4} ms | "
                f"L={r['left']:>6.2f} "
                f"R={r['right']:>6.2f} | "
                f"turn={r['turn']:>7.2f} "
                f"Δ={r['delta']:+7.2f}"
            )


if __name__ == "__main__":
    main()
