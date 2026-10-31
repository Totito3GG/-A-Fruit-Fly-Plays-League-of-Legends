import csv
from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime


WIDTH = 320
HEIGHT = 240

PIXEL = 188

TARGET_WIDTH = 64
TARGET_HEIGHT = 160

WARMUP_STEPS = 30

# 0.0125 * 320 ≈ 4 pixels per neural step.
STEP_X = 0.0125

X_MIN = 0.05
X_MAX = 0.95

STATIC_SCAN = Path(
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


def load_static_map():
    if not STATIC_SCAN.exists():
        raise FileNotFoundError(
            f"Missing {STATIC_SCAN}"
        )

    xs = []
    turns = []

    with STATIC_SCAN.open() as file:
        reader = csv.DictReader(file)

        for row in reader:
            xs.append(
                float(row["x"])
            )

            turns.append(
                float(row["delta_turn"])
            )

    return (
        np.asarray(xs),
        np.asarray(turns),
    )


def run_sweep(
    name,
    positions,
    static_x,
    static_turn,
):
    runtime = FlyQBrainRuntime()

    black = black_frame()

    baseline = []

    #
    # Identical dark warmup.
    #
    for _ in range(WARMUP_STEPS):
        result = runtime.process_frame(
            black
        )

        baseline.append(
            result["turn_raw_hz"]
        )

    baseline_turn = float(
        np.mean(
            baseline[-10:]
        )
    )

    results = []

    print()
    print(f"===== {name} =====")
    print()

    for step, x in enumerate(
        positions,
        start=1,
    ):
        frame = target_frame(x)

        result = runtime.process_frame(
            frame
        )

        turn = float(
            result["turn_raw_hz"]
        )

        movement = float(
            result["movement_raw_hz"]
        )

        dynamic_delta = (
            turn - baseline_turn
        )

        expected_static = float(
            np.interp(
                x,
                static_x,
                static_turn,
            )
        )

        residual = (
            dynamic_delta
            - expected_static
        )

        results.append({
            "x": float(x),
            "turn": turn,
            "dynamic_delta": dynamic_delta,
            "static_expected": expected_static,
            "residual": residual,
            "movement": movement,
        })

        #
        # Don't print every single frame.
        #
        if (
            step == 1
            or step % 8 == 0
            or step == len(positions)
        ):
            print(
                f"{step:>3} | "
                f"x={x:.3f} | "
                f"turn={turn:+7.2f} | "
                f"Δdynamic={dynamic_delta:+7.2f} | "
                f"static={expected_static:+7.2f} | "
                f"residual={residual:+7.2f} | "
                f"move={movement:7.2f}"
            )

    return results


def main():
    print()
    print(
        "FlyQ Localized Motion Residual Probe"
    )
    print(
        "===================================="
    )
    print()
    print(
        "Moving 64px target through actual RGB retina."
    )
    print(
        "Static retinotopy response is subtracted."
    )

    static_x, static_turn = (
        load_static_map()
    )

    rightward_positions = np.arange(
        X_MIN,
        X_MAX + STEP_X / 2,
        STEP_X,
    )

    leftward_positions = (
        rightward_positions[::-1]
    )

    rightward = run_sweep(
        "RIGHTWARD",
        rightward_positions,
        static_x,
        static_turn,
    )

    leftward = run_sweep(
        "LEFTWARD",
        leftward_positions,
        static_x,
        static_turn,
    )

    #
    # Compare the two directions at the SAME
    # screen positions.
    #
    right_by_x = {
        round(r["x"], 4): r
        for r in rightward
    }

    left_by_x = {
        round(r["x"], 4): r
        for r in leftward
    }

    differences = []

    interior = []

    for key in sorted(
        set(right_by_x)
        & set(left_by_x)
    ):
        x = float(key)

        #
        # Avoid edge clipping in the final summary.
        #
        if not (
            0.15 <= x <= 0.85
        ):
            continue

        right_residual = (
            right_by_x[key][
                "residual"
            ]
        )

        left_residual = (
            left_by_x[key][
                "residual"
            ]
        )

        difference = (
            right_residual
            - left_residual
        )

        differences.append(
            difference
        )

        interior.append((
            x,
            right_residual,
            left_residual,
            difference,
        ))

    differences = np.asarray(
        differences,
        dtype=float,
    )

    right_residuals = np.asarray([
        r["residual"]
        for r in rightward
        if 0.15 <= r["x"] <= 0.85
    ])

    left_residuals = np.asarray([
        r["residual"]
        for r in leftward
        if 0.15 <= r["x"] <= 0.85
    ])

    print()
    print("SAME-POSITION COMPARISON")
    print("========================")
    print()

    for (
        x,
        right_residual,
        left_residual,
        difference,
    ) in interior[::4]:

        print(
            f"x={x:.3f} | "
            f"RIGHTWARD residual="
            f"{right_residual:+7.2f} | "
            f"LEFTWARD residual="
            f"{left_residual:+7.2f} | "
            f"R-L={difference:+7.2f}"
        )

    print()
    print("SUMMARY")
    print("=======")
    print()

    print(
        "RIGHTWARD residual mean : "
        f"{right_residuals.mean():+.2f} Hz"
    )

    print(
        "LEFTWARD residual mean  : "
        f"{left_residuals.mean():+.2f} Hz"
    )

    print(
        "Direction separation    : "
        f"{differences.mean():+.2f} Hz"
    )

    print(
        "Separation std          : "
        f"{differences.std():.2f} Hz"
    )

    positive = int(
        np.sum(
            differences > 0
        )
    )

    negative = int(
        np.sum(
            differences < 0
        )
    )

    print(
        f"Same-position signs     : "
        f"{positive} positive / "
        f"{negative} negative"
    )


if __name__ == "__main__":
    main()
