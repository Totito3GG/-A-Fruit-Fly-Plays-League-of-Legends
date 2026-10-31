import csv
import gc
import math
from pathlib import Path

import numpy as np

from experiments.pixel_target_centering import (
    PixelController,
    PixelTurnDecoder,
    render_target,
    wrap_angle,
)

from experiments.pixel_target_centering_v2 import (
    PersistentTurnActuator,
)


# ============================================================
# World
# ============================================================

TARGET_DISTANCE = 0.80
TARGET_RADIUS = 0.12

TARGET_BEARINGS = [
    +45.0,
    +30.0,
    +15.0,
    -15.0,
    -30.0,
    -45.0,
]


# ============================================================
# Body
# ============================================================

TURN_STEP_DEG = 2.0
MOVE_STEP = 0.01


# ============================================================
# Sensorimotor sequencing
# ============================================================

#
# Neural steering must remain CENTER before
# requesting a forward burst.
#
CENTER_HOLD_STEPS = 5

#
# One forward burst moves the body this many
# actual movement steps.
#
DRIVE_MOVE_STEPS = 6

#
# Maximum neural steps allowed for DNpe017
# to generate one movement burst.
#
DRIVE_MAX_NEURAL_STEPS = 20

#
# After the movement cue, show target-only
# vision without acting so MaleCNS can
# transition back to the target scene.
#
RECOVERY_STEPS = 10


# ============================================================
# Movement readout
# ============================================================

MOVE_ON_HZ = 30.0
MOVE_OFF_HZ = 25.0

MOVE_CONFIRM = 3
STOP_CONFIRM = 2


# ============================================================
# Movement sensory pulse
# ============================================================

WIDTH = 320
HEIGHT = 240

MOVE_PIXEL = 188
MOVE_BAND_FRACTION = 0.30


# ============================================================
# Limits
# ============================================================

MAX_STEPS = 600

OUTPUT = Path(
    "telemetry/pixel_navigation_2d_v2.csv"
)


def target_xy(
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


def geometry(
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

    absolute_bearing = (
        math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )
    )

    relative_bearing = (
        wrap_angle(
            absolute_bearing
            - heading_deg
        )
    )

    return (
        distance,
        relative_bearing,
    )


def movement_cue_frame():
    """
    Engineered visual pulse that strongly
    activates DNpe017.

    It is NOT shown while steering.
    """

    frame = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )

    band = int(
        WIDTH
        * MOVE_BAND_FRACTION
    )

    frame[
        :,
        :band,
    ] = MOVE_PIXEL

    frame[
        :,
        WIDTH - band:,
    ] = MOVE_PIXEL

    return frame


class MovementBurstDecoder:
    def __init__(self):
        self.state = "STOP"

        self.high_count = 0
        self.low_count = 0

    def update(
        self,
        movement_hz,
    ):
        value = float(
            movement_hz
        )

        if self.state == "STOP":

            self.low_count = 0

            if value >= MOVE_ON_HZ:
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

            if value < MOVE_OFF_HZ:
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


def run_trial(
    trial_number,
    target_bearing,
):
    target = target_xy(
        target_bearing
    )

    position = [
        0.0,
        0.0,
    ]

    heading = 0.0

    controller = (
        PixelController()
    )

    turn_baseline = (
        controller.calibrate_black()
    )

    actuator = (
        PersistentTurnActuator()
    )

    movement_decoder = (
        MovementBurstDecoder()
    )

    phase = "SEEK"

    center_hold = 0

    drive_neural_steps = 0
    drive_move_steps = 0

    recovery_steps = 0

    telemetry = []

    min_distance = TARGET_DISTANCE

    success = False

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

    print()

    last_phase = None

    for step in range(
        1,
        MAX_STEPS + 1,
    ):
        (
            distance,
            relative_bearing,
        ) = geometry(
            position,
            heading,
            target,
        )

        min_distance = min(
            min_distance,
            distance,
        )

        #
        # SUCCESS CHECK = evaluation only.
        #
        if distance <= TARGET_RADIUS:
            success = True

            print()
            print(
                "TARGET REACHED ✅"
            )

            print(
                f"Step           : {step}"
            )

            print(
                f"Position       : "
                f"({position[0]:+.3f}, "
                f"{position[1]:+.3f})"
            )

            print(
                f"Heading        : "
                f"{heading:+.1f}°"
            )

            print(
                f"Distance       : "
                f"{distance:.3f}"
            )

            break

        target_x = None
        turn_action = "CENTER"
        turn_source = "NONE"
        movement_state = "STOP"

        neural_direction = "CENTER"
        filtered_turn = 0.0
        movement_hz = 0.0

        # ====================================================
        # SEEK
        # ====================================================

        if phase == "SEEK":

            (
                frame,
                target_x,
            ) = render_target(
                relative_bearing
            )

            neural = (
                controller.step(
                    frame
                )
            )

            neural_direction = (
                neural["direction"]
            )

            filtered_turn = (
                neural[
                    "filtered_delta_hz"
                ]
            )

            movement_hz = (
                neural[
                    "movement_hz"
                ]
            )

            (
                turn_action,
                turn_source,
            ) = actuator.update(
                neural_direction
            )

            if neural["ready"]:

                if (
                    turn_action
                    == "LEFT"
                ):
                    heading = wrap_angle(
                        heading
                        + TURN_STEP_DEG
                    )

                elif (
                    turn_action
                    == "RIGHT"
                ):
                    heading = wrap_angle(
                        heading
                        - TURN_STEP_DEG
                    )

            #
            # Important:
            # This uses ONLY the neural
            # steering decision.
            #
            if (
                neural["ready"]
                and turn_action
                == "CENTER"
            ):
                center_hold += 1

            else:
                center_hold = 0

            if (
                center_hold
                >= CENTER_HOLD_STEPS
            ):
                phase = "DRIVE"

                center_hold = 0

                drive_neural_steps = 0
                drive_move_steps = 0

                movement_decoder = (
                    MovementBurstDecoder()
                )

        # ====================================================
        # DRIVE
        # ====================================================

        elif phase == "DRIVE":

            frame = (
                movement_cue_frame()
            )

            neural = (
                controller.step(
                    frame
                )
            )

            movement_hz = float(
                neural[
                    "movement_hz"
                ]
            )

            movement_state = (
                movement_decoder.update(
                    movement_hz
                )
            )

            drive_neural_steps += 1

            if (
                movement_state
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

                drive_move_steps += 1

            if (
                drive_move_steps
                >= DRIVE_MOVE_STEPS
                or drive_neural_steps
                >= DRIVE_MAX_NEURAL_STEPS
            ):
                phase = "RECOVER"

                recovery_steps = 0

                #
                # Movement-cue DNp20 activity
                # must not remain in the
                # steering filter.
                #
                controller.decoder = (
                    PixelTurnDecoder()
                )

                actuator = (
                    PersistentTurnActuator()
                )

        # ====================================================
        # RECOVER / REACQUIRE
        # ====================================================

        elif phase == "RECOVER":

            (
                frame,
                target_x,
            ) = render_target(
                relative_bearing
            )

            neural = (
                controller.step(
                    frame
                )
            )

            neural_direction = (
                neural["direction"]
            )

            filtered_turn = (
                neural[
                    "filtered_delta_hz"
                ]
            )

            movement_hz = (
                neural[
                    "movement_hz"
                ]
            )

            #
            # No body action during recovery.
            #
            recovery_steps += 1

            if (
                recovery_steps
                >= RECOVERY_STEPS
            ):
                phase = "SEEK"

                center_hold = 0

                actuator = (
                    PersistentTurnActuator()
                )

        else:
            raise RuntimeError(
                f"Unknown phase: {phase}"
            )

        (
            distance_after,
            relative_after,
        ) = geometry(
            position,
            heading,
            target,
        )

        min_distance = min(
            min_distance,
            distance_after,
        )

        telemetry.append({
            "trial": trial_number,
            "step": step,

            "phase": phase,

            "target_bearing_start": (
                target_bearing
            ),

            "body_x": position[0],
            "body_y": position[1],

            "heading_deg": heading,

            "distance": (
                distance_after
            ),

            "relative_bearing_deg": (
                relative_after
            ),

            "target_screen_x": (
                ""
                if target_x is None
                else target_x
            ),

            "neural_direction": (
                neural_direction
            ),

            "turn_action": (
                turn_action
            ),

            "turn_source": (
                turn_source
            ),

            "filtered_turn_hz": (
                filtered_turn
            ),

            "movement_hz": (
                movement_hz
            ),

            "movement_state": (
                movement_state
            ),

            "drive_move_steps": (
                drive_move_steps
            ),
        })

        #
        # Compact console output.
        #
        if (
            phase != last_phase
            or step % 20 == 0
            or distance_after
            <= 0.20
        ):
            x_text = (
                "OFF"
                if target_x is None
                else f"{target_x:.3f}"
            )

            print(
                f"{step:>3} | "
                f"phase={phase:>7} | "
                f"x={x_text:>5} | "
                f"turn={turn_action:>6} | "
                f"move={movement_state:>4} | "
                f"DNp20Δ="
                f"{filtered_turn:+7.2f} | "
                f"DNpe017="
                f"{movement_hz:6.2f} | "
                f"pos=("
                f"{position[0]:+.3f},"
                f"{position[1]:+.3f}) | "
                f"head="
                f"{heading:+6.1f}° | "
                f"dist="
                f"{distance_after:.3f}"
            )

        last_phase = phase

    if not success:

        (
            distance,
            relative,
        ) = geometry(
            position,
            heading,
            target,
        )

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

        "steps": (
            step
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
        "FlyQ Pixel-Driven 2D Navigation V2"
    )

    print(
        "=================================="
    )

    print()
    print(
        "Neural sensorimotor sequencing:"
    )

    print(
        "  SEEK    : RGB target -> DNp20"
    )

    print(
        "  DRIVE   : RGB movement cue -> DNpe017"
    )

    print(
        "  RECOVER : RGB target, no action"
    )

    print()

    print(
        "No target coordinates are passed "
        "to the controller."
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
            result["telemetry"]
        )

        successes += int(
            result["success"]
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
            if result["success"]
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
