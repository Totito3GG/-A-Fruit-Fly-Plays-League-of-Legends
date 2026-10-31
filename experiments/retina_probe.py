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
STIMULATION_MS = 500


def find_readout(readouts, neuron_type, side):
    for r in readouts:
        if r["type"] == neuron_type and r["side"] == side:
            return r

    raise RuntimeError(f"Missing {neuron_type} {side}")


def make_light(brain, condition, intensity):
    uv = brain.uv
    x = uv[:, 0]

    light = np.zeros(len(x), dtype=np.float32)

    if condition == "black":
        return light, np.array([], dtype=int)

    if condition == "left":
        indices = np.argsort(x)[:ACTIVE_RECEPTORS]

    elif condition == "right":
        indices = np.argsort(x)[-ACTIVE_RECEPTORS:]

    elif condition == "center":
        indices = np.argsort(np.abs(x - 0.5))[:ACTIVE_RECEPTORS]

    else:
        raise ValueError(condition)

    light[indices] = intensity

    return light, indices


def run(condition, intensity, manifest):
    brain = NativeBrain(GRAPH)

    controls = NeuralControls(
        manifest["readouts"],
        mode="bci",
    )

    light, indices = make_light(
        brain,
        condition,
        intensity,
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

    dnp20_l = find_readout(readouts, "DNp20", "L")
    dnp20_r = find_readout(readouts, "DNp20", "R")

    dnpe_l = find_readout(readouts, "DNpe017", "L")
    dnpe_r = find_readout(readouts, "DNpe017", "R")

    result = {
        "condition": condition,
        "intensity": intensity,
        "active": len(indices),

        "dnp20_l": float(dnp20_l["rate_hz"]),
        "dnp20_r": float(dnp20_r["rate_hz"]),

        "turn": float(
            dnp20_r["rate_hz"]
            - dnp20_l["rate_hz"]
        ),

        "dnpe_l": float(dnpe_l["rate_hz"]),
        "dnpe_r": float(dnpe_r["rate_hz"]),

        "move": float(
            dnpe_l["rate_hz"]
            + dnpe_r["rate_hz"]
        ),

        "doom_turn": float(action["turn"]),
        "doom_forward": float(action["forward"]),
        "doom_attack": bool(action["attack"]),

        "wall": float(wall),
    }

    del brain
    gc.collect()

    return result


def main():
    manifest = json.loads(
        MANIFEST.read_text()
    )

    print()
    print("FlyQ Equal-Energy Retina Probe")
    print("==============================")
    print(f"Active receptors per stimulus: {ACTIVE_RECEPTORS}")
    print()

    results = []

    # Establish the zero-input baseline.
    results.append(
        run("black", 0.0, manifest)
    )

    for intensity in [0.25, 0.50, 1.00]:
        for condition in [
            "left",
            "center",
            "right",
        ]:
            print(
                f"Running {condition:>6} "
                f"intensity={intensity:.2f} ...",
                flush=True,
            )

            results.append(
                run(
                    condition,
                    intensity,
                    manifest,
                )
            )

    baseline = results[0]["turn"]

    print()
    print("RESULTS")
    print("=======")

    for r in results:
        delta = r["turn"] - baseline

        print(
            f"{r['condition']:>6} "
            f"I={r['intensity']:.2f} "
            f"N={r['active']:>4} | "
            f"DNp20 "
            f"L={r['dnp20_l']:6.2f} "
            f"R={r['dnp20_r']:6.2f} | "
            f"turn={r['turn']:7.2f} "
            f"Δ={delta:+7.2f} | "
            f"move={r['move']:7.2f}"
        )


if __name__ == "__main__":
    main()
