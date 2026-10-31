import gc
import json
import sys
from pathlib import Path

import numpy as np


UPSTREAM = Path.home() / "FlyQ" / "upstream-doomfly"

if str(UPSTREAM) not in sys.path:
    sys.path.insert(
        0,
        str(UPSTREAM),
    )

from doom.native import NativeBrain
from doom.engine import NeuralControls
from doom.game import retinal_samples


GRAPH = (
    UPSTREAM
    / "outputs"
    / "doom"
    / "malecns_v1"
    / "graph.npz"
)

MANIFEST = (
    UPSTREAM
    / "outputs"
    / "doom"
    / "malecns_v1"
    / "manifest.json"
)


WIDTH = 320
HEIGHT = 240

PIXEL = 188

TARGET_WIDTH = 64
TARGET_HEIGHT = 160

STEP_MS = 28.6

WARMUP_STEPS = 20
STIMULUS_STEPS = 20
MEASURE_STEPS = 7

#
# This ±3 Hz threshold is ONLY for this
# engineering comparison.
#
CLASS_THRESHOLD = 3.0


def black_frame():
    return np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )


def left_half_frame():
    frame = black_frame()

    frame[
        :,
        : WIDTH // 2,
    ] = PIXEL

    return frame


def right_half_frame():
    frame = black_frame()

    frame[
        :,
        WIDTH // 2 :,
    ] = PIXEL

    return frame


def center_third_frame():
    frame = black_frame()

    x1 = WIDTH // 3
    x2 = 2 * WIDTH // 3

    frame[
        :,
        x1:x2,
    ] = PIXEL

    return frame


def target_frame(x_fraction):
    frame = black_frame()

    cx = int(
        x_fraction
        * (WIDTH - 1)
    )

    cy = HEIGHT // 2

    x1 = max(
        0,
        cx - TARGET_WIDTH // 2,
    )

    x2 = min(
        WIDTH,
        cx + TARGET_WIDTH // 2,
    )

    y1 = max(
        0,
        cy - TARGET_HEIGHT // 2,
    )

    y2 = min(
        HEIGHT,
        cy + TARGET_HEIGHT // 2,
    )

    frame[
        y1:y2,
        x1:x2,
    ] = PIXEL

    return frame


def find_readout(
    readouts,
    neuron_type,
    side,
):
    for readout in readouts:
        if (
            readout.get("type")
            == neuron_type
            and
            readout.get("side")
            == side
        ):
            return readout

    raise RuntimeError(
        f"Missing readout: "
        f"{neuron_type} side={side}"
    )


def get_pair(readouts, neuron_type):
    left = find_readout(
        readouts,
        neuron_type,
        "L",
    )

    right = find_readout(
        readouts,
        neuron_type,
        "R",
    )

    left_hz = float(
        left["rate_hz"]
    )

    right_hz = float(
        right["rate_hz"]
    )

    return (
        left_hz,
        right_hz,
        right_hz - left_hz,
    )


def step_brain(
    brain,
    controls,
    light,
):
    counts, _ = brain.step(
        light,
        STEP_MS,
    )

    action = controls.decode(
        counts,
        STEP_MS / 1000.0,
    )

    readouts = action[
        "readouts"
    ]

    dnp_l, dnp_r, dnp_diff = (
        get_pair(
            readouts,
            "DNp20",
        )
    )

    dna_l, dna_r, dna_diff = (
        get_pair(
            readouts,
            "DNa02",
        )
    )

    return {
        "dnp_l": dnp_l,
        "dnp_r": dnp_r,
        "dnp": dnp_diff,

        "dna_l": dna_l,
        "dna_r": dna_r,
        "dna": dna_diff,
    }


def classify(value):
    if value < -CLASS_THRESHOLD:
        return "LEFT"

    if value > CLASS_THRESHOLD:
        return "RIGHT"

    return "CENTER"


def run_condition(
    name,
    frame,
    expected,
):
    manifest = json.loads(
        MANIFEST.read_text()
    )

    brain = NativeBrain(
        GRAPH
    )

    controls = NeuralControls(
        manifest["readouts"],
        mode="bci",
    )

    black = black_frame()

    black_light = retinal_samples(
        black,
        brain.uv,
    )

    stimulus_light = retinal_samples(
        frame,
        brain.uv,
    )

    baseline_dnp = []
    baseline_dna = []

    #
    # Same dark warmup for every
    # condition.
    #
    for _ in range(
        WARMUP_STEPS
    ):
        result = step_brain(
            brain,
            controls,
            black_light,
        )

        baseline_dnp.append(
            result["dnp"]
        )

        baseline_dna.append(
            result["dna"]
        )

    baseline_dnp = float(
        np.mean(
            baseline_dnp[
                -MEASURE_STEPS:
            ]
        )
    )

    baseline_dna = float(
        np.mean(
            baseline_dna[
                -MEASURE_STEPS:
            ]
        )
    )

    dnp_values = []
    dna_values = []

    dnp_l_values = []
    dnp_r_values = []

    dna_l_values = []
    dna_r_values = []

    for _ in range(
        STIMULUS_STEPS
    ):
        result = step_brain(
            brain,
            controls,
            stimulus_light,
        )

        dnp_values.append(
            result["dnp"]
        )

        dna_values.append(
            result["dna"]
        )

        dnp_l_values.append(
            result["dnp_l"]
        )

        dnp_r_values.append(
            result["dnp_r"]
        )

        dna_l_values.append(
            result["dna_l"]
        )

        dna_r_values.append(
            result["dna_r"]
        )

    tail = slice(
        -MEASURE_STEPS,
        None,
    )

    dnp_mean = float(
        np.mean(
            dnp_values[tail]
        )
    )

    dna_mean = float(
        np.mean(
            dna_values[tail]
        )
    )

    dnp_delta = (
        dnp_mean
        - baseline_dnp
    )

    dna_delta = (
        dna_mean
        - baseline_dna
    )

    dnp_std = float(
        np.std(
            dnp_values[tail]
        )
    )

    dna_std = float(
        np.std(
            dna_values[tail]
        )
    )

    result = {
        "name": name,
        "expected": expected,

        "retina": float(
            stimulus_light.mean()
        ),

        "dnp_delta": dnp_delta,
        "dnp_std": dnp_std,

        "dna_delta": dna_delta,
        "dna_std": dna_std,

        "dna_flipped": (
            -dna_delta
        ),

        "dnp_state": classify(
            dnp_delta
        ),

        "dna_state": classify(
            dna_delta
        ),

        "dna_flipped_state": classify(
            -dna_delta
        ),

        "dnp_l": float(
            np.mean(
                dnp_l_values[tail]
            )
        ),

        "dnp_r": float(
            np.mean(
                dnp_r_values[tail]
            )
        ),

        "dna_l": float(
            np.mean(
                dna_l_values[tail]
            )
        ),

        "dna_r": float(
            np.mean(
                dna_r_values[tail]
            )
        ),
    }

    #
    # Important on the 8 GB WSL machine:
    # release the full brain before
    # constructing the next condition.
    #
    del brain
    del controls

    gc.collect()

    return result


def correct(
    observed,
    expected,
):
    return (
        observed == expected
    )


def main():
    conditions = [
        (
            "black",
            black_frame(),
            "CENTER",
        ),

        (
            "left_half",
            left_half_frame(),
            "LEFT",
        ),

        (
            "center_third",
            center_third_frame(),
            "CENTER",
        ),

        (
            "right_half",
            right_half_frame(),
            "RIGHT",
        ),

        (
            "target_x10",
            target_frame(0.10),
            "LEFT",
        ),

        (
            "target_x30",
            target_frame(0.30),
            "LEFT",
        ),

        (
            "target_x50",
            target_frame(0.50),
            "CENTER",
        ),

        (
            "target_x70",
            target_frame(0.70),
            "RIGHT",
        ),

        (
            "target_x90",
            target_frame(0.90),
            "RIGHT",
        ),
    ]

    print()
    print(
        "FlyQ Turn Readout Benchmark"
    )
    print(
        "=========================="
    )

    print()
    print(
        "Comparing:"
    )

    print(
        "  DNp20 : R - L"
    )

    print(
        "  DNa02 : R - L"
    )

    print()
    print(
        "Fresh brain per condition."
    )

    print(
        "Same RGB -> retina -> "
        "MaleCNS stimulus."
    )

    print()

    results = []

    for (
        name,
        frame,
        expected,
    ) in conditions:

        print(
            f"Running {name}..."
        )

        result = run_condition(
            name,
            frame,
            expected,
        )

        results.append(
            result
        )

    print()
    print(
        "RESULTS — R-L convention"
    )
    print(
        "========================"
    )
    print()

    for r in results:
        dnp_ok = correct(
            r["dnp_state"],
            r["expected"],
        )

        dna_ok = correct(
            r["dna_state"],
            r["expected"],
        )

        print(
            f"{r['name']:>12} | "
            f"expected={r['expected']:>6} | "
            f"retina={r['retina']:.3f} | "
            f"DNp20={r['dnp_delta']:+7.2f} "
            f"±{r['dnp_std']:.2f} "
            f"{r['dnp_state']:>6} "
            f"{'✅' if dnp_ok else '❌'} | "
            f"DNa02={r['dna_delta']:+7.2f} "
            f"±{r['dna_std']:.2f} "
            f"{r['dna_state']:>6} "
            f"{'✅' if dna_ok else '❌'}"
        )

    print()
    print(
        "DNa02 POLARITY CHECK"
    )
    print(
        "===================="
    )
    print()

    print(
        "We also test L-R in case "
        "DNa02 has the opposite sign."
    )

    print()

    for r in results:
        state = (
            r[
                "dna_flipped_state"
            ]
        )

        ok = correct(
            state,
            r["expected"],
        )

        print(
            f"{r['name']:>12} | "
            f"DNa02 L-R="
            f"{r['dna_flipped']:+7.2f} | "
            f"{state:>6} | "
            f"{'✅' if ok else '❌'}"
        )

    dnp_correct = sum(
        correct(
            r["dnp_state"],
            r["expected"],
        )
        for r in results
    )

    dna_correct = sum(
        correct(
            r["dna_state"],
            r["expected"],
        )
        for r in results
    )

    dna_flip_correct = sum(
        correct(
            r[
                "dna_flipped_state"
            ],
            r["expected"],
        )
        for r in results
    )

    #
    # Useful continuous comparisons.
    #
    by_name = {
        r["name"]: r
        for r in results
    }

    dnp_half_sep = (
        by_name["right_half"][
            "dnp_delta"
        ]
        - by_name["left_half"][
            "dnp_delta"
        ]
    )

    dna_half_sep = (
        by_name["right_half"][
            "dna_delta"
        ]
        - by_name["left_half"][
            "dna_delta"
        ]
    )

    dnp_target_sep = (
        np.mean([
            by_name["target_x70"][
                "dnp_delta"
            ],
            by_name["target_x90"][
                "dnp_delta"
            ],
        ])
        -
        np.mean([
            by_name["target_x10"][
                "dnp_delta"
            ],
            by_name["target_x30"][
                "dnp_delta"
            ],
        ])
    )

    dna_target_sep = (
        np.mean([
            by_name["target_x70"][
                "dna_delta"
            ],
            by_name["target_x90"][
                "dna_delta"
            ],
        ])
        -
        np.mean([
            by_name["target_x10"][
                "dna_delta"
            ],
            by_name["target_x30"][
                "dna_delta"
            ],
        ])
    )

    print()
    print(
        "SUMMARY"
    )
    print(
        "======="
    )
    print()

    print(
        f"DNp20 R-L classifications : "
        f"{dnp_correct}/"
        f"{len(results)}"
    )

    print(
        f"DNa02 R-L classifications : "
        f"{dna_correct}/"
        f"{len(results)}"
    )

    print(
        f"DNa02 L-R classifications : "
        f"{dna_flip_correct}/"
        f"{len(results)}"
    )

    print()

    print(
        "Half-field separation:"
    )

    print(
        f"  DNp20 : "
        f"{dnp_half_sep:+.2f} Hz"
    )

    print(
        f"  DNa02 : "
        f"{dna_half_sep:+.2f} Hz"
    )

    print()

    print(
        "Localized-target "
        "right-minus-left separation:"
    )

    print(
        f"  DNp20 : "
        f"{dnp_target_sep:+.2f} Hz"
    )

    print(
        f"  DNa02 : "
        f"{dna_target_sep:+.2f} Hz"
    )

    print()
    print(
        "CENTER BIAS"
    )
    print(
        "==========="
    )

    print(
        f"DNp20 target x=.50 : "
        f"{by_name['target_x50']['dnp_delta']:+.2f}"
    )

    print(
        f"DNa02 target x=.50 : "
        f"{by_name['target_x50']['dna_delta']:+.2f}"
    )

    print()
    print(
        "PAIR RATES"
    )
    print(
        "=========="
    )

    print()

    for r in results:
        print(
            f"{r['name']:>12} | "
            f"DNp20 "
            f"L={r['dnp_l']:6.2f} "
            f"R={r['dnp_r']:6.2f} | "
            f"DNa02 "
            f"L={r['dna_l']:6.2f} "
            f"R={r['dna_r']:6.2f}"
        )


if __name__ == "__main__":
    main()
