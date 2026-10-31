import csv
from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime


WIDTH = 320
HEIGHT = 240

PIXEL = 188

TARGET_WIDTH = 64
TARGET_HEIGHT = 160

WARMUP_STEPS = 20
STIMULUS_STEPS = 20
MEASURE_STEPS = 7

X_POSITIONS = np.arange(
    0.05,
    0.951,
    0.05,
)

OUTPUT = Path(
    "telemetry/rgb_retinotopy_scan.csv"
)


def black_frame():
    return np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )


def target_frame(x_fraction):
    frame = black_frame()

    cx = int(
        x_fraction * (WIDTH - 1)
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


def run_position(x_fraction):
    #
    # Fresh brain for every position.
    # This prevents previous stimuli from
    # contaminating the spatial map.
    #
    runtime = FlyQBrainRuntime()

    black = black_frame()

    target = target_frame(
        x_fraction
    )

    baseline_turn = []
    baseline_move = []

    for _ in range(
        WARMUP_STEPS
    ):
        result = runtime.process_frame(
            black
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
    retina = []

    for _ in range(
        STIMULUS_STEPS
    ):
        result = runtime.process_frame(
            target
        )

        turns.append(
            result["turn_raw_hz"]
        )

        movements.append(
            result["movement_raw_hz"]
        )

        retina.append(
            result["retina_mean"]
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

    retina_mean = float(
        np.mean(
            retina[
                -MEASURE_STEPS:
            ]
        )
    )

    delta_turn = (
        turn - baseline_turn
    )

    delta_move = (
        movement - baseline_move
    )

    if delta_turn < -3.0:
        steering = "LEFT"

    elif delta_turn > 3.0:
        steering = "RIGHT"

    else:
        steering = "CENTER"

    return {
        "x": float(x_fraction),

        "retina": retina_mean,

        "baseline_turn": baseline_turn,
        "turn": turn,
        "delta_turn": delta_turn,

        "baseline_move": baseline_move,
        "movement": movement,
        "delta_move": delta_move,

        "steering": steering,
    }


def bar(value):
    count = int(
        min(
            abs(value) / 2,
            20,
        )
    )

    if value < 0:
        return (
            "<" * count
        ).rjust(20) + "|"

    return (
        " " * 20
        + "|"
        + ">" * count
    )


def main():
    print()
    print(
        "FlyQ RGB Retinotopy Scan"
    )
    print(
        "========================"
    )

    print()
    print(
        f"Target: {TARGET_WIDTH}x"
        f"{TARGET_HEIGHT}px"
    )

    print()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for x in X_POSITIONS:

        print(
            f"Scanning x={x:.2f}..."
        )

        result = run_position(
            x
        )

        results.append(
            result
        )

    print()
    print("RESULTS")
    print("=======")
    print()

    for r in results:

        print(
            f"x={r['x']:.2f} | "
            f"retina={r['retina']:.3f} | "
            f"Δturn={r['delta_turn']:+7.2f} | "
            f"{r['steering']:>6} | "
            f"Δmove={r['delta_move']:+7.2f} | "
            f"{bar(r['delta_turn'])}"
        )

    with OUTPUT.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print()
    print(
        f"Saved → {OUTPUT}"
    )


if __name__ == "__main__":
    main()
