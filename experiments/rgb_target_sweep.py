import numpy as np

from neural.runtime import FlyQBrainRuntime


WIDTH = 320
HEIGHT = 240

PIXEL = 188

WARMUP_STEPS = 20
STIMULUS_STEPS = 20
MEASURE_STEPS = 7

TARGET_HEIGHT = 160

TARGET_WIDTHS = [
    32,
    64,
    96,
]

X_POSITIONS = [
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


def target_frame(
    x_fraction,
    target_width,
):
    frame = black_frame()

    cx = int(
        x_fraction * (WIDTH - 1)
    )

    cy = HEIGHT // 2

    x1 = max(
        0,
        cx - target_width // 2,
    )

    x2 = min(
        WIDTH,
        cx + target_width // 2,
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


def run(
    x_fraction,
    target_width,
):
    runtime = FlyQBrainRuntime()

    black = black_frame()

    target = target_frame(
        x_fraction,
        target_width,
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

    baseline_turn_mean = float(
        np.mean(
            baseline_turn[
                -MEASURE_STEPS:
            ]
        )
    )

    baseline_move_mean = float(
        np.mean(
            baseline_move[
                -MEASURE_STEPS:
            ]
        )
    )

    turns = []
    movement = []
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

        movement.append(
            result["movement_raw_hz"]
        )

        retina.append(
            result["retina_mean"]
        )

    turn_mean = float(
        np.mean(
            turns[
                -MEASURE_STEPS:
            ]
        )
    )

    move_mean = float(
        np.mean(
            movement[
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

    turn_shift = (
        turn_mean
        - baseline_turn_mean
    )

    move_shift = (
        move_mean
        - baseline_move_mean
    )

    if x_fraction < 0.4:
        expected = "LEFT"

    elif x_fraction > 0.6:
        expected = "RIGHT"

    else:
        expected = "CENTER"

    if turn_shift < -3.0:
        observed = "LEFT"

    elif turn_shift > 3.0:
        observed = "RIGHT"

    else:
        observed = "CENTER"

    correct = (
        observed == expected
    )

    return {
        "x": x_fraction,
        "width": target_width,

        "retina": retina_mean,

        "turn": turn_mean,
        "turn_shift": turn_shift,

        "movement": move_mean,
        "movement_shift": move_shift,

        "expected": expected,
        "observed": observed,

        "correct": correct,
    }


def main():
    print()
    print(
        "FlyQ RGB Local Target Sweep"
    )
    print(
        "============================"
    )
    print()

    print(
        "Testing localized visible "
        "objects through:"
    )

    print(
        "RGB -> retina -> MaleCNS"
    )

    print()

    results = []

    for width in TARGET_WIDTHS:

        print()
        print(
            f"===== WIDTH {width}px ====="
        )
        print()

        for x in X_POSITIONS:

            result = run(
                x,
                width,
            )

            results.append(
                result
            )

            print(
                f"x={x:.2f} | "
                f"retina={result['retina']:.3f} | "
                f"turn={result['turn']:+7.2f} | "
                f"Δturn={result['turn_shift']:+7.2f} | "
                f"move={result['movement']:7.2f} | "
                f"Δmove={result['movement_shift']:+7.2f} | "
                f"expected={result['expected']:>6} | "
                f"observed={result['observed']:>6} | "
                f"{'✅' if result['correct'] else '❌'}"
            )

    print()
    print(
        "SIZE SUMMARY"
    )
    print(
        "============"
    )
    print()

    for width in TARGET_WIDTHS:

        subset = [
            r
            for r in results
            if r["width"] == width
        ]

        correct = sum(
            r["correct"]
            for r in subset
        )

        print(
            f"{width:>3}px : "
            f"{correct}/{len(subset)} "
            f"positions classified correctly"
        )


if __name__ == "__main__":
    main()
