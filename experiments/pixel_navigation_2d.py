import csv
import gc
import math
from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime

from experiments.pixel_target_centering import (
    PixelTurnDecoder,
    wrap_angle,
)

from experiments.pixel_target_centering_v2 import (
    PersistentTurnActuator,
)


# ============================================================
# Camera / scene
# ============================================================

WIDTH = 320
HEIGHT = 240

FOV_DEG = 120.0
HALF_FOV_DEG = FOV_DEG / 2.0

#
# Engineered constant peripheral visual context.
#
# Earlier RGB probes showed broad peripheral bilateral
# stimulation strongly drives DNpe017.
#
BACKGROUND_PIXEL = 188
BACKGROUND_FRACTION = 0.30

#
# Target is brighter than environment.
#
TARGET_PIXEL = 255
TARGET_WIDTH = 64
TARGET_HEIGHT = 160


# ============================================================
# Body
# ============================================================

TURN_STEP_DEG = 2.0
MOVE_STEP = 0.01

TARGET_RADIUS = 0.12

MAX_STEPS = 450

WARMUP_STEPS = 30


# ============================================================
# DNpe017 movement decoder
# ============================================================

MOVE_ON_HZ = 30.0
MOVE_OFF_HZ = 25.0

MOVE_CONFIRM = 3
STOP_CONFIRM = 2


# ============================================================
# Trials
# ============================================================

TARGET_DISTANCE = 0.80

TARGET_BEARINGS = [
    +45.0,
    +30.0,
    +15.0,
    -15.0,
    -30.0,
    -45.0,
]


OUTPUT = Path(
    "telemetry/pixel_navigation_2d.csv"
)


def background_frame():
    frame = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )

    band = int(
        WIDTH * BACKGROUND_FRACTION
    )

    frame[
        :,
        :band,
    ] = BACKGROUND_PIXEL

    frame[
        :,
        WIDTH - band:,
    ] = BACKGROUND_PIXEL

    return frame


def target_xy_from_bearing(
    bearing_deg,
):
    angle = math.radians(
        bearing_deg
    )

    return (
        TARGET_DISTANCE
        * math.cos(angle),

        TARGET_DISTANCE
        * math.sin(angle),
    )


def world_geometry(
    position,
    heading_deg,
    target,
):
    dx = (
        target[0]
        - position[0]
    )

    dy = (
        target[1]
        - position[1]
    )

    distance = math.hypot(
        dx,
        dy,
    )

    absolute_bearing = math.degrees(
        math.atan2(
            dy,
            dx,
        )
    )

    relative_bearing = wrap_angle(
        absolute_bearing
        - heading_deg
    )

    return (
        distance,
        absolute_bearing,
        relative_bearing,
    )


def render_scene(
    position,
    heading_deg,
    target,
):
    """
    WORLD / RENDERER.

    This function knows target coordinates because
    it must render the world.

    The neural controller receives only the returned
    RGB image.
    """

    frame = background_frame()

    (
        distance,
        absolute_bearing,
        relative_bearing,
    ) = world_geometry(
        position,
        heading_deg,
        target,
    )

    if (
        abs(relative_bearing)
        > HALF_FOV_DEG
    ):
        return (
            frame,
            None,
            distance,
            relative_bearing,
        )

    #
    # Positive bearing = target to the left.
    #
    x_fraction = (
        0.5
        - relative_bearing
        / FOV_DEG
    )

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
    ] = TARGET_PIXEL

    return (
        frame,
        x_fraction,
        distance,
        relative_bearing,
    )


class ForwardDecoder:
    """
    DNpe017 movement decoder.

    Same engineering idea as our previous
    movement decoder:

        STOP -> MOVE:
            3 consecutive >= 30 Hz

        MOVE -> STOP:
            2 consecutive < 25 Hz
    """

    def __init__(self):
        self.state = "STOP"

        self.high_count = 0
        self.low_count = 0

    def update(
        self,
        movement_hz,
    ):
        movement_hz = float(
            movement_hz
        )

        if self.state == "STOP":
            self.low_count = 0

            if (
                movement_hz
                >= MOVE_ON_HZ
            ):
                self.high_count += 1
            else:
                self.high_count = 0

            if (
                self.high_count
                >= MOVE_CONFIRM
            ):
                self.state = "MOVE"

                self.high_count = 0

        else:
            self.high_count = 0

            if (
                movement_hz
                < MOVE_OFF_HZ
            ):
                self.low_count += 1
            else:
                self.low_count = 0

            if (
                self.low_count
                >= STOP_CONFIRM
            ):
                self.state = "STOP"

                self.low_count = 0

        return self.state


class PixelNavigationController:
    """
    Controller input = RGB frame only.

    Outputs:
        turn action from DNp20
        forward state from DNpe017
    """

    def __init__(self):
        self.runtime = (
            FlyQBrainRuntime()
        )

        self.turn_decoder = (
            PixelTurnDecoder()
        )

        self.turn_actuator = (
            PersistentTurnActuator()
        )

        self.forward_decoder = (
            ForwardDecoder()
        )

        self.turn_baseline = None

    def calibrate_environment(
        self,
    ):
        """
        Calibrate DNp20 with the constant
        peripheral environment present but
        NO target.

        Body does not move during calibration.
        """

        frame = background_frame()

        turns = []
        moves = []

        for _ in range(
            WARMUP_STEPS
        ):
            result = (
                self.runtime
                .process_frame(
                    frame
                )
            )

            turns.append(
                float(
                    result[
                        "turn_raw_hz"
                    ]
                )
            )

            moves.append(
                float(
                    result[
                        "movement_raw_hz"
                    ]
                )
            )

        self.turn_baseline = float(
            np.mean(
                turns[-10:]
            )
        )

        movement_baseline = float(
            np.mean(
                moves[-10:]
            )
        )

        #
        # Start behavioral decoders fresh
        # after neural warmup.
        #
        self.turn_decoder = (
            PixelTurnDecoder()
        )

        self.turn_actuator = (
            PersistentTurnActuator()
        )

        self.forward_decoder = (
            ForwardDecoder()
        )

        return (
            self.turn_baseline,
            movement_baseline,
        )

    def step(
        self,
        frame,
    ):
        result = (
            self.runtime
            .process_frame(
                frame
            )
        )

        raw_turn = float(
            result[
                "turn_raw_hz"
            ]
        )

        movement_hz = float(
            result[
                "movement_raw_hz"
            ]
        )

        turn_delta = (
            raw_turn
            - self.turn_baseline
        )

        (
            neural_direction,
            filtered_turn,
            ready,
        ) = (
            self.turn_decoder
            .update(
                turn_delta
            )
        )

        (
            turn_action,
            turn_source,
        ) = (
            self.turn_actuator
            .update(
                neural_direction
            )
        )

        forward_state = (
            self.forward_decoder
            .update(
                movement_hz
            )
        )

        return {
            "ready": ready,

            "neural_direction": (
                neural_direction
            ),

            "turn_action": (
                turn_action
            ),

            "turn_source": (
                turn_source
            ),

            "forward": (
                forward_state
            ),

            "raw_turn_hz": (
                raw_turn
            ),

            "turn_delta_hz": (
                turn_delta
            ),

            "filtered_turn_hz": (
                filtered_turn
            ),

            "movement_hz": (
                movement_hz
            ),
        }


def run_trial(
    trial_number,
    target_bearing,
):
    target = (
        target_xy_from_bearing(
            target_bearing
        )
    )

    position = [
        0.0,
        0.0,
    ]

    heading = 0.0

    controller = (
        PixelNavigationController()
    )

    (
        turn_baseline,
        movement_baseline,
    ) = (
        controller
        .calibrate_environment()
    )

    print()
    print(
        f"===== TRIAL {trial_number} ====="
    )

    print(
        f"Target bearing : "
        f"{target_bearing:+.1f}°"
    )

    print(
        f"Target position: "
        f"({target[0]:+.3f}, "
        f"{target[1]:+.3f})"
    )

    print(
        f"Turn baseline  : "
        f"{turn_baseline:+.2f} Hz"
    )

    print(
        f"Movement env   : "
        f"{movement_baseline:.2f} Hz"
    )

    print()

    telemetry = []

    success = False

    previous_turn = None
    previous_forward = None

    min_distance = float(
        "inf"
    )

    for step in range(
        1,
        MAX_STEPS + 1,
    ):
        (
            frame,
            target_x,
            distance_before,
            bearing_before,
        ) = render_scene(
            position,
            heading,
            target,
        )

        #
        # CONTROLLER:
        #
        # RGB pixels only.
        #
        neural = (
            controller.step(
                frame
            )
        )

        #
        # TURN BODY
        #
        if neural["ready"]:
            if (
                neural["turn_action"]
                == "LEFT"
            ):
                heading = wrap_angle(
                    heading
                    + TURN_STEP_DEG
                )

            elif (
                neural["turn_action"]
                == "RIGHT"
            ):
                heading = wrap_angle(
                    heading
                    - TURN_STEP_DEG
                )

        #
        # MOVE BODY
        #
        if (
            neural["ready"]
            and neural["forward"]
            == "MOVE"
        ):
            heading_rad = (
                math.radians(
                    heading
                )
            )

            position[0] += (
                MOVE_STEP
                * math.cos(
                    heading_rad
                )
            )

            position[1] += (
                MOVE_STEP
                * math.sin(
                    heading_rad
                )
            )

        #
        # EVALUATION ONLY.
        #
        (
            distance,
            absolute_bearing,
            relative_bearing,
        ) = world_geometry(
            position,
            heading,
            target,
        )

        min_distance = min(
            min_distance,
            distance,
        )

        visible = (
            target_x
            is not None
        )

        telemetry.append({
            "trial": trial_number,
            "step": step,

            "target_bearing_start": (
                target_bearing
            ),

            "target_x_world": (
                target[0]
            ),

            "target_y_world": (
                target[1]
            ),

            "body_x": (
                position[0]
            ),

            "body_y": (
                position[1]
            ),

            "heading_deg": (
                heading
            ),

            "distance": (
                distance
            ),

            "relative_bearing_deg": (
                relative_bearing
            ),

            "visible": int(
                visible
            ),

            "target_screen_x": (
                ""
                if target_x is None
                else target_x
            ),

            "neural_direction": (
                neural[
                    "neural_direction"
                ]
            ),

            "turn_action": (
                neural[
                    "turn_action"
                ]
            ),

            "turn_source": (
                neural[
                    "turn_source"
                ]
            ),

            "forward": (
                neural[
                    "forward"
                ]
            ),

            "turn_delta_hz": (
                neural[
                    "turn_delta_hz"
                ]
            ),

            "filtered_turn_hz": (
                neural[
                    "filtered_turn_hz"
                ]
            ),

            "movement_hz": (
                neural[
                    "movement_hz"
                ]
            ),
        })

        should_print = (
            step == 1
            or step % 10 == 0
            or neural[
                "turn_action"
            ] != previous_turn
            or neural[
                "forward"
            ] != previous_forward
            or distance
            <= 0.20
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
                f"turn="
                f"{neural['turn_action']:>6} | "
                f"fwd="
                f"{neural['forward']:>4} | "
                f"DNp20Δ="
                f"{neural['filtered_turn_hz']:+7.2f} | "
                f"DNpe017="
                f"{neural['movement_hz']:6.2f} | "
                f"pos=("
                f"{position[0]:+.3f},"
                f"{position[1]:+.3f}) | "
                f"head="
                f"{heading:+6.1f}° | "
                f"dist="
                f"{distance:.3f}"
            )

        previous_turn = (
            neural[
                "turn_action"
            ]
        )

        previous_forward = (
            neural[
                "forward"
            ]
        )

        if (
            distance
            <= TARGET_RADIUS
        ):
            success = True

            print()
            print(
                "TARGET REACHED ✅ "
                f"after {step} neural steps"
            )

            print(
                f"Final position : "
                f"({position[0]:+.3f}, "
                f"{position[1]:+.3f})"
            )

            print(
                f"Final heading  : "
                f"{heading:+.1f}°"
            )

            print(
                f"Final distance : "
                f"{distance:.3f}"
            )

            break

    if not success:
        print()
        print(
            "TARGET NOT REACHED ❌"
        )

        print(
            f"Final position : "
            f"({position[0]:+.3f}, "
            f"{position[1]:+.3f})"
        )

        print(
            f"Final heading  : "
            f"{heading:+.1f}°"
        )

        print(
            f"Final distance : "
            f"{distance:.3f}"
        )

        print(
            f"Minimum distance: "
            f"{min_distance:.3f}"
        )

    del controller
    gc.collect()

    return {
        "success": success,
        "steps": len(
            telemetry
        ),

        "position": tuple(
            position
        ),

        "heading": heading,

        "distance": distance,

        "min_distance": (
            min_distance
        ),

        "telemetry": telemetry,
    }


def main():
    print()
    print(
        "FlyQ Pixel-Driven 2D Navigation"
    )
    print(
        "================================"
    )

    print()
    print(
        "Controller receives RGB pixels ONLY."
    )

    print(
        "Steering : DNp20 R-L"
    )

    print(
        "Movement : DNpe017 L+R"
    )

    print(
        "Success  : enter target radius"
    )

    print()

    print(
        f"Target radius : "
        f"{TARGET_RADIUS:.2f}"
    )

    print(
        f"Move step     : "
        f"{MOVE_STEP:.3f}"
    )

    print(
        f"Turn step     : "
        f"{TURN_STEP_DEG:.1f}°"
    )

    print()

    results = []

    all_rows = []

    successes = 0

    for trial, bearing in enumerate(
        TARGET_BEARINGS,
        start=1,
    ):
        result = run_trial(
            trial,
            bearing,
        )

        results.append(
            (
                bearing,
                result,
            )
        )

        all_rows.extend(
            result[
                "telemetry"
            ]
        )

        successes += int(
            result[
                "success"
            ]
        )

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
                all_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            all_rows
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

    for trial, (
        bearing,
        result,
    ) in enumerate(
        results,
        start=1,
    ):
        status = (
            "✅"
            if result[
                "success"
            ]
            else "❌"
        )

        print(
            f"Trial {trial} | "
            f"target="
            f"{bearing:+6.1f}° | "
            f"steps="
            f"{result['steps']:>3} | "
            f"final_dist="
            f"{result['distance']:.3f} | "
            f"min_dist="
            f"{result['min_distance']:.3f} | "
            f"{status}"
        )

    print()

    print(
        f"Targets reached: "
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
