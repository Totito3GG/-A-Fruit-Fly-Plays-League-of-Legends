import csv
import gc
import math
from collections import deque
from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime


# ============================================================
# Camera
# ============================================================

WIDTH = 320
HEIGHT = 240

PIXEL = 188

TARGET_WIDTH = 64
TARGET_HEIGHT = 160

FOV_DEG = 120.0
HALF_FOV_DEG = FOV_DEG / 2.0


# ============================================================
# Neural controller
# ============================================================

#
# Same ~200 ms integration we used before:
#
# 7 × 28.6 ms ≈ 200.2 ms
#
WINDOW_SIZE = 7

#
# These are ENGINEERING thresholds for the localized
# RGB target, based on our benchmark.
#
# They are not biological thresholds.
#
LEFT_THRESHOLD = -3.0
RIGHT_THRESHOLD = +7.0


# ============================================================
# Body
# ============================================================

TURN_STEP_DEG = 2.0

TARGET_ANGLE_TOLERANCE = 8.0

STABLE_STEPS = 5

MAX_STEPS = 250

WARMUP_STEPS = 30


# ============================================================
# Trials
# ============================================================

#
# Positive angle = target is LEFT of heading.
# Negative angle = target is RIGHT of heading.
#
TARGET_BEARINGS = [
    +45.0,
    +30.0,
    +15.0,
    -15.0,
    -30.0,
    -45.0,
]

OUTPUT = Path(
    "telemetry/pixel_target_centering.csv"
)


def wrap_angle(angle):
    while angle > 180.0:
        angle -= 360.0

    while angle <= -180.0:
        angle += 360.0

    return angle


def black_frame():
    return np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )


def render_target(relative_angle):
    """
    World/camera side only.

    This function knows the target angle because a renderer
    must know where an object is in order to draw it.

    The neural controller never receives relative_angle.
    """

    frame = black_frame()

    if abs(relative_angle) > HALF_FOV_DEG:
        return frame, None

    #
    # Positive angle = target left of body.
    #
    # Therefore:
    #
    #   +60 deg -> x=0.0
    #     0 deg -> x=0.5
    #   -60 deg -> x=1.0
    #
    x_fraction = (
        0.5
        - relative_angle / FOV_DEG
    )

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

    return frame, x_fraction


class PixelTurnDecoder:
    """
    Pixel-target steering decoder.

    Input:
        baseline-corrected DNp20 R-L only.

    It knows NOTHING about:
        - target coordinates
        - target bearing
        - target x position
        - desired direction
    """

    def __init__(self):
        self.values = deque(
            maxlen=WINDOW_SIZE
        )

    def update(self, turn_delta):
        self.values.append(
            float(turn_delta)
        )

        if len(self.values) < WINDOW_SIZE:
            return (
                "CENTER",
                float(
                    np.mean(
                        self.values
                    )
                ),
                False,
            )

        filtered = float(
            np.mean(
                self.values
            )
        )

        if filtered < LEFT_THRESHOLD:
            direction = "LEFT"

        elif filtered > RIGHT_THRESHOLD:
            direction = "RIGHT"

        else:
            direction = "CENTER"

        return (
            direction,
            filtered,
            True,
        )


class PixelController:
    """
    FlyQ controller.

    Important:
    step() receives ONLY an RGB frame.
    """

    def __init__(self):
        self.runtime = FlyQBrainRuntime()

        self.baseline_turn = None

        self.decoder = PixelTurnDecoder()

    def calibrate_black(self):
        frame = black_frame()

        values = []

        for _ in range(
            WARMUP_STEPS
        ):
            result = (
                self.runtime.process_frame(
                    frame
                )
            )

            values.append(
                result["turn_raw_hz"]
            )

        #
        # Use the final 10 warmup states as
        # the dark baseline.
        #
        self.baseline_turn = float(
            np.mean(
                values[-10:]
            )
        )

        #
        # Decoder starts fresh after
        # baseline calibration.
        #
        self.decoder = PixelTurnDecoder()

        return self.baseline_turn

    def step(self, frame):
        result = (
            self.runtime.process_frame(
                frame
            )
        )

        raw_turn = float(
            result["turn_raw_hz"]
        )

        turn_delta = (
            raw_turn
            - self.baseline_turn
        )

        (
            direction,
            filtered_delta,
            ready,
        ) = self.decoder.update(
            turn_delta
        )

        return {
            "direction": direction,

            "ready": ready,

            "raw_turn_hz": raw_turn,

            "turn_delta_hz": (
                turn_delta
            ),

            "filtered_delta_hz": (
                filtered_delta
            ),

            "movement_hz": float(
                result[
                    "movement_raw_hz"
                ]
            ),
        }


def run_trial(
    trial_number,
    target_bearing,
):
    #
    # Fresh brain for each independent
    # visual-centering trial.
    #
    controller = PixelController()

    baseline = (
        controller.calibrate_black()
    )

    heading = 0.0

    stable = 0

    telemetry = []

    previous_direction = None

    print()
    print(
        f"===== TRIAL {trial_number} ====="
    )

    print(
        f"Target bearing : "
        f"{target_bearing:+.1f}°"
    )

    print(
        f"Initial heading: "
        f"{heading:+.1f}°"
    )

    print(
        f"Black baseline : "
        f"{baseline:+.2f} Hz"
    )

    print()

    acquired = False

    for step in range(
        1,
        MAX_STEPS + 1,
    ):
        #
        # WORLD / RENDERER
        #
        relative_angle = wrap_angle(
            target_bearing
            - heading
        )

        (
            frame,
            target_x,
        ) = render_target(
            relative_angle
        )

        #
        # CONTROLLER
        #
        # This receives PIXELS ONLY.
        #
        neural = controller.step(
            frame
        )

        direction = neural[
            "direction"
        ]

        #
        # BODY
        #
        if neural["ready"]:

            if direction == "LEFT":
                heading = wrap_angle(
                    heading
                    + TURN_STEP_DEG
                )

            elif direction == "RIGHT":
                heading = wrap_angle(
                    heading
                    - TURN_STEP_DEG
                )

        #
        # Recalculate only for evaluation
        # after the body action.
        #
        evaluation_error = wrap_angle(
            target_bearing
            - heading
        )

        #
        # Scoring only.
        #
        # This does NOT command the body.
        #
        centered = (
            abs(evaluation_error)
            <= TARGET_ANGLE_TOLERANCE
        )

        if (
            centered
            and neural["ready"]
            and direction == "CENTER"
        ):
            stable += 1

        else:
            stable = 0

        visible = (
            target_x is not None
        )

        telemetry.append({
            "trial": trial_number,
            "step": step,

            "target_bearing_deg": (
                target_bearing
            ),

            "heading_deg": heading,

            "relative_angle_deg": (
                evaluation_error
            ),

            "target_x": (
                ""
                if target_x is None
                else target_x
            ),

            "visible": int(
                visible
            ),

            "direction": direction,

            "ready": int(
                neural["ready"]
            ),

            "raw_turn_hz": (
                neural[
                    "raw_turn_hz"
                ]
            ),

            "turn_delta_hz": (
                neural[
                    "turn_delta_hz"
                ]
            ),

            "filtered_delta_hz": (
                neural[
                    "filtered_delta_hz"
                ]
            ),

            "movement_hz": (
                neural[
                    "movement_hz"
                ]
            ),

            "stable": stable,
        })

        #
        # Console output:
        # state changes, every 10 steps,
        # near center, or invisible target.
        #
        should_print = (
            step == 1
            or step % 10 == 0
            or direction
            != previous_direction
            or abs(
                evaluation_error
            ) <= 12.0
            or not visible
        )

        if should_print:

            x_text = (
                "OFF"
                if target_x is None
                else f"{target_x:.3f}"
            )

            print(
                f"{step:>3} | "
                f"x={x_text:>5} | "
                f"dir={direction:>6} | "
                f"rawΔ="
                f"{neural['turn_delta_hz']:+7.2f} | "
                f"filtΔ="
                f"{neural['filtered_delta_hz']:+7.2f} | "
                f"head={heading:+7.1f}° | "
                f"err="
                f"{evaluation_error:+7.1f}° | "
                f"stable={stable}"
            )

        previous_direction = (
            direction
        )

        if stable >= STABLE_STEPS:
            acquired = True

            print()
            print(
                "TARGET CENTERED ✅ "
                f"after {step} neural steps"
            )

            print(
                f"Final heading : "
                f"{heading:+.1f}°"
            )

            print(
                f"Final error   : "
                f"{evaluation_error:+.1f}°"
            )

            break

    if not acquired:
        print()
        print(
            "TARGET NOT CENTERED ❌"
        )

        print(
            f"Final heading : "
            f"{heading:+.1f}°"
        )

        print(
            f"Final error   : "
            f"{evaluation_error:+.1f}°"
        )

    #
    # Explicitly release the large
    # whole-brain runtime before the
    # next trial.
    #
    del controller

    gc.collect()

    return (
        acquired,
        heading,
        evaluation_error,
        telemetry,
    )


def main():
    print()
    print(
        "FlyQ Pixel-Only Target Centering"
    )

    print(
        "================================"
    )

    print()
    print(
        "Controller input: RGB pixels ONLY"
    )

    print(
        "Turn signal     : DNp20 R-L"
    )

    print(
        f"Camera FOV      : {FOV_DEG:.1f}°"
    )

    print(
        f"Turn step       : "
        f"{TURN_STEP_DEG:.1f}°"
    )

    print(
        f"Center tolerance: "
        f"±{TARGET_ANGLE_TOLERANCE:.1f}°"
    )

    print(
        f"Decoder window  : "
        f"{WINDOW_SIZE} neural steps"
    )

    print(
        f"Thresholds      : "
        f"LEFT < {LEFT_THRESHOLD:+.1f}, "
        f"RIGHT > {RIGHT_THRESHOLD:+.1f} Hz"
    )

    all_telemetry = []

    successes = 0

    summary = []

    for trial_number, bearing in enumerate(
        TARGET_BEARINGS,
        start=1,
    ):
        (
            acquired,
            heading,
            error,
            telemetry,
        ) = run_trial(
            trial_number,
            bearing,
        )

        successes += int(
            acquired
        )

        all_telemetry.extend(
            telemetry
        )

        summary.append({
            "bearing": bearing,
            "acquired": acquired,
            "heading": heading,
            "error": error,
            "steps": len(
                telemetry
            ),
        })

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                all_telemetry[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            all_telemetry
        )

    print()
    print(
        "=============================="
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=============================="
    )

    print()

    for i, item in enumerate(
        summary,
        start=1,
    ):

        status = (
            "✅"
            if item["acquired"]
            else "❌"
        )

        print(
            f"Trial {i} | "
            f"target="
            f"{item['bearing']:+6.1f}° | "
            f"steps="
            f"{item['steps']:>3} | "
            f"final="
            f"{item['heading']:+6.1f}° | "
            f"error="
            f"{item['error']:+6.1f}° | "
            f"{status}"
        )

    print()

    print(
        f"Targets centered: "
        f"{successes}/"
        f"{len(TARGET_BEARINGS)}"
    )

    print()

    print(
        f"Telemetry saved → "
        f"{OUTPUT}"
    )


if __name__ == "__main__":
    main()
