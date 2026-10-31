import sys
from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime


UPSTREAM = (
    Path.home()
    / "FlyQ"
    / "upstream-doomfly"
)

if str(UPSTREAM) not in sys.path:
    sys.path.insert(
        0,
        str(UPSTREAM),
    )

from doom.game import retinal_samples


DATASET = "malecns_v1"

GRAPH = (
    UPSTREAM
    / "outputs"
    / "doom"
    / DATASET
    / "graph.npz"
)


WIDTH = 320
HEIGHT = 240

PIXEL = 188

TARGET_WIDTH = 64
TARGET_HEIGHT = 160

WARMUP_STEPS = 20
STIMULUS_STEPS = 20
MEASURE_STEPS = 7

BINS = 10

POSITIONS = [
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
]


def black_frame():
    return np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )


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


def build_density_weights(uv):
    x = np.asarray(
        uv[:, 0],
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        BINS + 1,
    )

    #
    # Receptors receive a horizontal
    # bin ID from 0 .. BINS-1.
    #
    bin_ids = np.digitize(
        x,
        edges[1:-1],
    )

    counts = np.bincount(
        bin_ids,
        minlength=BINS,
    )

    nonzero = counts[
        counts > 0
    ]

    #
    # Use the sparsest bin as the
    # reference density.
    #
    # Therefore weights are <= 1.
    # We attenuate dense regions but
    # never amplify sparse ones.
    #
    reference = int(
        nonzero.min()
    )

    weights = np.ones(
        len(x),
        dtype=np.float32,
    )

    for bin_id in range(BINS):

        count = counts[bin_id]

        if count == 0:
            continue

        weights[
            bin_ids == bin_id
        ] = (
            reference
            / float(count)
        )

    return (
        weights,
        counts,
        edges,
        reference,
    )


def classify(delta_turn):
    if delta_turn < -3.0:
        return "LEFT"

    if delta_turn > 3.0:
        return "RIGHT"

    return "CENTER"


def expected_direction(x):
    if x < 0.4:
        return "LEFT"

    if x > 0.6:
        return "RIGHT"

    return "CENTER"


def run_condition(
    x_fraction,
    uv,
    weights,
    compensated,
):
    runtime = FlyQBrainRuntime()

    #
    # Sanity check:
    # receptor order must match the
    # graph used to build the weights.
    #
    if not np.allclose(
        runtime.brain.uv,
        uv,
    ):
        raise RuntimeError(
            "Runtime UV order differs "
            "from graph UV order."
        )

    black = black_frame()

    stimulus = target_frame(
        x_fraction
    )

    black_light = retinal_samples(
        black,
        uv,
    )

    stimulus_light = retinal_samples(
        stimulus,
        uv,
    )

    if compensated:
        black_light = (
            black_light
            * weights
        )

        stimulus_light = (
            stimulus_light
            * weights
        )

    baseline_turn = []
    baseline_move = []

    for _ in range(
        WARMUP_STEPS
    ):
        result = runtime.process_light(
            black_light
        )

        baseline_turn.append(
            result["turn_raw_hz"]
        )

        baseline_move.append(
            result["movement_raw_hz"]
        )

    baseline_turn = float(
        np.mean(
            baseline_turn[
                -MEASURE_STEPS:
            ]
        )
    )

    baseline_move = float(
        np.mean(
            baseline_move[
                -MEASURE_STEPS:
            ]
        )
    )

    turns = []
    movements = []

    for _ in range(
        STIMULUS_STEPS
    ):
        result = runtime.process_light(
            stimulus_light
        )

        turns.append(
            result["turn_raw_hz"]
        )

        movements.append(
            result["movement_raw_hz"]
        )

    turn = float(
        np.mean(
            turns[
                -MEASURE_STEPS:
            ]
        )
    )

    movement = float(
        np.mean(
            movements[
                -MEASURE_STEPS:
            ]
        )
    )

    delta_turn = (
        turn
        - baseline_turn
    )

    delta_move = (
        movement
        - baseline_move
    )

    return {
        "turn": turn,
        "delta_turn": delta_turn,

        "movement": movement,
        "delta_move": delta_move,

        "observed": classify(
            delta_turn
        ),

        "retina_energy": float(
            stimulus_light.sum()
        ),
    }


def main():
    graph = np.load(
        GRAPH,
        allow_pickle=False,
    )

    uv = np.asarray(
        graph["uv"],
        dtype=np.float32,
    )

    graph.close()

    (
        weights,
        counts,
        edges,
        reference,
    ) = build_density_weights(
        uv
    )

    print()
    print(
        "FlyQ Density-Compensated Retinotopy"
    )
    print(
        "==================================="
    )

    print()
    print(
        "Horizontal density compensation"
    )
    print(
        "--------------------------------"
    )

    print(
        f"Reference count: {reference}"
    )

    print()

    for i, count in enumerate(counts):

        weight = float(
            reference
            / count
        )

        print(
            f"{edges[i]:.1f}-"
            f"{edges[i+1]:.1f} | "
            f"N={count:>4} | "
            f"gain={weight:.3f}"
        )

    print()
    print(
        "A/B RETINOTOPY"
    )
    print(
        "=============="
    )
    print()

    original_correct = 0
    compensated_correct = 0

    for x in POSITIONS:

        original = run_condition(
            x,
            uv,
            weights,
            compensated=False,
        )

        compensated = run_condition(
            x,
            uv,
            weights,
            compensated=True,
        )

        expected = expected_direction(
            x
        )

        original_ok = (
            original["observed"]
            == expected
        )

        compensated_ok = (
            compensated["observed"]
            == expected
        )

        original_correct += int(
            original_ok
        )

        compensated_correct += int(
            compensated_ok
        )

        print(
            f"x={x:.2f} "
            f"expected={expected:>6}"
        )

        print(
            f"  ORIGINAL    | "
            f"Δturn="
            f"{original['delta_turn']:+7.2f} | "
            f"{original['observed']:>6} | "
            f"Δmove="
            f"{original['delta_move']:+7.2f} | "
            f"energy="
            f"{original['retina_energy']:7.2f} | "
            f"{'✅' if original_ok else '❌'}"
        )

        print(
            f"  COMPENSATED | "
            f"Δturn="
            f"{compensated['delta_turn']:+7.2f} | "
            f"{compensated['observed']:>6} | "
            f"Δmove="
            f"{compensated['delta_move']:+7.2f} | "
            f"energy="
            f"{compensated['retina_energy']:7.2f} | "
            f"{'✅' if compensated_ok else '❌'}"
        )

        print()

    print(
        "SUMMARY"
    )
    print(
        "======="
    )

    print(
        f"Original    : "
        f"{original_correct}/"
        f"{len(POSITIONS)} correct"
    )

    print(
        f"Compensated : "
        f"{compensated_correct}/"
        f"{len(POSITIONS)} correct"
    )

    print()

    if compensated_correct > original_correct:
        print(
            "Density compensation improved "
            "static visual steering."
        )

    elif compensated_correct == original_correct:
        print(
            "Density compensation did not "
            "improve classification count."
        )

    else:
        print(
            "Density compensation made "
            "classification worse."
        )


if __name__ == "__main__":
    main()
