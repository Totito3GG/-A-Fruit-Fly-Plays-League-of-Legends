from pathlib import Path
import json
import sys

import numpy as np


UPSTREAM = Path.home() / "FlyQ" / "upstream-doomfly"
sys.path.insert(0, str(UPSTREAM))

from doom.native import NativeBrain
from doom.game import retinal_samples
from doom.engine import NeuralControls


GRAPH = UPSTREAM / "outputs/doom/malecns_v1/graph.npz"
MANIFEST = UPSTREAM / "outputs/doom/malecns_v1/manifest.json"


WIDTH = 320
HEIGHT = 240


def make_frame(condition: str) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    if condition == "black":
        pass

    elif condition == "white":
        frame[:] = 255

    elif condition == "left":
        frame[:, : WIDTH // 2] = 255

    elif condition == "right":
        frame[:, WIDTH // 2 :] = 255

    elif condition == "center":
        start = WIDTH // 3
        end = 2 * WIDTH // 3
        frame[:, start:end] = 255

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return frame


def find_readout(readouts, neuron_type, side):
    for r in readouts:
        if r["type"] == neuron_type and r["side"] == side:
            return r

    raise RuntimeError(f"Missing {neuron_type} {side}")


def run_condition(condition, manifest):
    # Fresh brain for every condition.
    # This prevents the previous stimulus from affecting the next trial.
    brain = NativeBrain(GRAPH)

    controls = NeuralControls(
        manifest["readouts"],
        mode="bci",
    )

    frame = make_frame(condition)

    light = retinal_samples(frame, brain.uv)

    # 500 ms controlled stimulation, matching DoomFly's validation approach.
    counts, wall = brain.step(light, 500)

    action = controls.decode(counts, 0.5)
    readouts = action["readouts"]

    dnp20_l = find_readout(readouts, "DNp20", "L")
    dnp20_r = find_readout(readouts, "DNp20", "R")

    dnpe017_l = find_readout(readouts, "DNpe017", "L")
    dnpe017_r = find_readout(readouts, "DNpe017", "R")

    return {
        "condition": condition,
        "retina_mean": float(light.mean()),
        "retina_active": int(np.count_nonzero(light)),
        "DNp20_L": float(dnp20_l["rate_hz"]),
        "DNp20_R": float(dnp20_r["rate_hz"]),
        "turn_raw": float(dnp20_r["rate_hz"] - dnp20_l["rate_hz"]),
        "DNpe017_L": float(dnpe017_l["rate_hz"]),
        "DNpe017_R": float(dnpe017_r["rate_hz"]),
        "move_raw": float(
            dnpe017_l["rate_hz"] + dnpe017_r["rate_hz"]
        ),
        "doom_turn": float(action["turn"]),
        "doom_forward": float(action["forward"]),
        "doom_attack": bool(action["attack"]),
        "wall_seconds": float(wall),
    }


def main():
    manifest = json.loads(MANIFEST.read_text())

    conditions = [
        "black",
        "left",
        "center",
        "right",
        "white",
    ]

    print()
    print("FlyQ Controlled Visual Probe")
    print("============================")
    print()

    results = []

    for condition in conditions:
        print(f"Running: {condition} ...", flush=True)

        result = run_condition(condition, manifest)
        results.append(result)

        print(
            f"{condition:>6} | "
            f"retina={result['retina_mean']:.3f} | "
            f"DNp20 L={result['DNp20_L']:6.2f} "
            f"R={result['DNp20_R']:6.2f} "
            f"diff={result['turn_raw']:7.2f} | "
            f"DNpe017 L={result['DNpe017_L']:6.2f} "
            f"R={result['DNpe017_R']:6.2f} "
            f"sum={result['move_raw']:7.2f}"
        )

    print()
    print("Summary")
    print("=======")

    for r in results:
        print(
            f"{r['condition']:>6}: "
            f"turn={r['turn_raw']:7.2f} Hz, "
            f"move={r['move_raw']:7.2f} Hz, "
            f"Doom=({r['doom_turn']:.3f}, "
            f"{r['doom_forward']:.3f}, "
            f"attack={r['doom_attack']})"
        )


if __name__ == "__main__":
    main()
