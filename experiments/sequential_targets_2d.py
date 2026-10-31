import csv
import gc
import math
import random
from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime

from experiments.pixel_target_centering import (
    PixelTurnDecoder,
    render_target,
    wrap_angle,
)

from experiments.pixel_target_centering_v2 import (
    PersistentTurnActuator,
)

from experiments.pixel_navigation_2d_v2 import (
    MovementBurstDecoder,
    movement_cue_frame,
)


# ============================================================
# Experiment
# ============================================================

SEED = 20260913

N_TARGETS = 10

TARGET_RADIUS = 0.12

#
# Each new target starts between:
#
#     10° .. 50°
#
# from current heading.
#
# Five left, five right.
#
MIN_BEARING = 10.0
MAX_BEARING = 50.0

#
# Each target is placed this far from
# FlyQ's CURRENT position.
#
MIN_DISTANCE = 0.55
MAX_DISTANCE = 0.95


# ============================================================
# Body
# ============================================================

TURN_STEP_DEG = 2.0
MOVE_STEP = 0.01


# ============================================================
# V2 sensorimotor policy
# ============================================================

CENTER_HOLD_STEPS = 5

DRIVE_MOVE_STEPS = 6
DRIVE_MAX_NEURAL_STEPS = 20

RECOVERY_STEPS = 10

WARMUP_STEPS = 30


# ============================================================
# Limits
# ============================================================

MAX_STEPS_PER_TARGET = 700

OUTPUT = Path(
    "telemetry/sequential_targets_2d.csv"
)


def generate_target_specs():
    """
    Reproducible but mixed target sequence.

    Five begin to the LEFT.
    Five begin to the RIGHT.

    Distances are also randomized.
    """

    rng = random.Random(
        SEED
    )

    signs = (
        [-1] * 5
        + [1] * 5
    )

    rng.shuffle(
        signs
    )

    specs = []

    for sign in signs:

        magnitude = rng.uniform(
            MIN_BEARING,
            MAX_BEARING,
        )

        relative_bearing = (
            sign
            * magnitude
        )

        distance = rng.uniform(
            MIN_DISTANCE,
            MAX_DISTANCE,
        )

        specs.append({
            "relative_bearing": (
                relative_bearing
            ),
            "distance": distance,
        })

    return specs


def make_world_target(
    position,
    heading_deg,
    relative_bearing,
    distance,
):
    """
    WORLD ONLY.

    The controller never receives the resulting
    coordinates or relative bearing.
    """

    absolute_bearing = wrap_angle(
        heading_deg
        + relative_bearing
    )

    radians = math.radians(
        absolute_bearing
    )

    return (
        position[0]
        + distance
        * math.cos(radians),

        position[1]
        + distance
        * math.sin(radians),
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

    relative_bearing = wrap_angle(
        absolute_bearing
        - heading_deg
    )

    return (
        distance,
        relative_bearing,
    )


class PersistentFlyQ:
    """
    ONE MaleCNS runtime for the ENTIRE experiment.

    This object is created once.

    It is never recreated between targets.
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

        self.turn_baseline = None

    def calibrate_black(self):

        frame = np.zeros(
            (240, 320, 3),
            dtype=np.uint8,
        )

        values = []

        for _ in range(
            WARMUP_STEPS
        ):
            result = (
                self.runtime
                .process_frame(
                    frame
                )
            )

            values.append(
                float(
                    result[
                        "turn_raw_hz"
                    ]
                )
            )

        self.turn_baseline = float(
            np.mean(
                values[-10:]
            )
        )

        #
        # Clean engineering filters after
        # calibration.
        #
        # MaleCNS itself is NOT reset.
        #
        self.turn_decoder = (
            PixelTurnDecoder()
        )

        self.turn_actuator = (
            PersistentTurnActuator()
        )

        return self.turn_baseline

    def seek(
        self,
        frame,
    ):
        """
        RGB target → MaleCNS → DNp20.
        """

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

        delta = (
            raw_turn
            - self.turn_baseline
        )

        (
            direction,
            filtered,
            ready,
        ) = (
            self.turn_decoder
            .update(
                delta
            )
        )

        (
            action,
            source,
        ) = (
            self.turn_actuator
            .update(
                direction
            )
        )

        return {
            "direction": direction,
            "action": action,
            "source": source,
            "filtered": filtered,
            "ready": ready,
            "movement_hz": movement_hz,
        }

    def observe_target(
        self,
        frame,
    ):
        """
        Recovery observation.

        MaleCNS keeps evolving.
        DNp20 filter keeps receiving target vision.

        Body does NOT act.
        """

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

        delta = (
            raw_turn
            - self.turn_baseline
        )

        (
            direction,
            filtered,
            ready,
        ) = (
            self.turn_decoder
            .update(
                delta
            )
        )

        return {
            "direction": direction,
            "filtered": filtered,
            "ready": ready,
            "movement_hz": movement_hz,
        }

    def movement_step(
        self,
        frame,
    ):
        """
        Movement cue → MaleCNS → DNpe017.

        Important:
        We DO NOT feed this movement image into
        the DNp20 engineering filter.

        MaleCNS still sees it.
        """

        result = (
            self.runtime
            .process_frame(
                frame
            )
        )

        return float(
            result[
                "movement_raw_hz"
            ]
        )

    def clear_motor_command(self):
        """
        Clear only our engineered turn actuator.

        MaleCNS neural state remains untouched.
        """

        self.turn_actuator = (
            PersistentTurnActuator()
        )


def main():

    specs = (
        generate_target_specs()
    )

    print()
    print(
        "FlyQ Sequential 2D Targets"
    )

    print(
        "=========================="
    )

    print()
    print(
        "10 targets"
    )

    print(
        "ONE persistent MaleCNS brain"
    )

    print(
        "ONE continuous body trajectory"
    )

    print(
        "NO neural reset between targets"
    )

    print()

    print(
        f"Random seed: {SEED}"
    )

    print()

    #
    # --------------------------------------------------------
    # Create FlyQ ONCE.
    # --------------------------------------------------------
    #

    flyq = PersistentFlyQ()

    baseline = (
        flyq.calibrate_black()
    )

    print(
        f"Black DNp20 baseline: "
        f"{baseline:+.2f} Hz"
    )

    print()

    position = [
        0.0,
        0.0,
    ]

    heading = 0.0

    global_step = 0

    rows = []
    summaries = []

    successes = 0

    total_path_length = 0.0

    for target_index, spec in enumerate(
        specs,
        start=1,
    ):

        initial_position = tuple(
            position
        )

        initial_heading = heading

        initial_relative = (
            spec[
                "relative_bearing"
            ]
        )

        initial_distance = (
            spec[
                "distance"
            ]
        )

        target = (
            make_world_target(
                position,
                heading,
                initial_relative,
                initial_distance,
            )
        )

        print()
        print(
            "================================"
        )

        print(
            f"TARGET {target_index}/{N_TARGETS}"
        )

        print(
            "================================"
        )

        print(
            f"Initial body : "
            f"({position[0]:+.3f}, "
            f"{position[1]:+.3f})"
        )

        print(
            f"Heading      : "
            f"{heading:+.1f}°"
        )

        print(
            f"New target relative bearing: "
            f"{initial_relative:+.1f}°"
        )

        print(
            f"New target distance        : "
            f"{initial_distance:.3f}"
        )

        print(
            f"World target : "
            f"({target[0]:+.3f}, "
            f"{target[1]:+.3f})"
        )

        print()

        #
        # Target-specific behavioral state.
        #
        # MaleCNS is NOT touched.
        #

        if target_index == 1:
            phase = "SEEK"
        else:
            #
            # New target appears.
            # Give persistent MaleCNS a short
            # target-view reacquisition period.
            #
            phase = "RECOVER"

        center_hold = 0

        recovery_steps = 0

        drive_neural_steps = 0
        drive_move_steps = 0

        movement_decoder = (
            MovementBurstDecoder()
        )

        flyq.clear_motor_command()

        reached = False

        min_distance = float(
            "inf"
        )

        target_steps = 0

        previous_phase = None

        for _ in range(
            MAX_STEPS_PER_TARGET
        ):

            global_step += 1
            target_steps += 1

            (
                distance_before,
                relative_before,
            ) = geometry(
                position,
                heading,
                target,
            )

            min_distance = min(
                min_distance,
                distance_before,
            )

            target_x = None

            neural_direction = (
                "CENTER"
            )

            turn_action = (
                "CENTER"
            )

            turn_source = (
                "NONE"
            )

            filtered_turn = 0.0

            movement_hz = 0.0

            movement_state = (
                "STOP"
            )

            # ================================================
            # SEEK
            # ================================================

            if phase == "SEEK":

                (
                    frame,
                    target_x,
                ) = render_target(
                    relative_before
                )

                neural = flyq.seek(
                    frame
                )

                neural_direction = (
                    neural[
                        "direction"
                    ]
                )

                turn_action = (
                    neural[
                        "action"
                    ]
                )

                turn_source = (
                    neural[
                        "source"
                    ]
                )

                filtered_turn = (
                    neural[
                        "filtered"
                    ]
                )

                movement_hz = (
                    neural[
                        "movement_hz"
                    ]
                )

                if neural["ready"]:

                    if (
                        turn_action
                        == "LEFT"
                    ):

                        heading = (
                            wrap_angle(
                                heading
                                + TURN_STEP_DEG
                            )
                        )

                    elif (
                        turn_action
                        == "RIGHT"
                    ):

                        heading = (
                            wrap_angle(
                                heading
                                - TURN_STEP_DEG
                            )
                        )

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

            # ================================================
            # DRIVE
            # ================================================

            elif phase == "DRIVE":

                frame = (
                    movement_cue_frame()
                )

                movement_hz = (
                    flyq.movement_step(
                        frame
                    )
                )

                movement_state = (
                    movement_decoder
                    .update(
                        movement_hz
                    )
                )

                drive_neural_steps += 1

                if (
                    movement_state
                    == "MOVE"
                ):

                    radians = (
                        math.radians(
                            heading
                        )
                    )

                    dx = (
                        MOVE_STEP
                        * math.cos(
                            radians
                        )
                    )

                    dy = (
                        MOVE_STEP
                        * math.sin(
                            radians
                        )
                    )

                    position[0] += dx
                    position[1] += dy

                    total_path_length += (
                        math.hypot(
                            dx,
                            dy,
                        )
                    )

                    drive_move_steps += 1

                if (
                    drive_move_steps
                    >= DRIVE_MOVE_STEPS
                    or
                    drive_neural_steps
                    >= DRIVE_MAX_NEURAL_STEPS
                ):

                    phase = "RECOVER"

                    recovery_steps = 0

                    flyq.clear_motor_command()

            # ================================================
            # RECOVER
            # ================================================

            elif phase == "RECOVER":

                (
                    frame,
                    target_x,
                ) = render_target(
                    relative_before
                )

                neural = (
                    flyq.observe_target(
                        frame
                    )
                )

                neural_direction = (
                    neural[
                        "direction"
                    ]
                )

                filtered_turn = (
                    neural[
                        "filtered"
                    ]
                )

                movement_hz = (
                    neural[
                        "movement_hz"
                    ]
                )

                recovery_steps += 1

                if (
                    recovery_steps
                    >= RECOVERY_STEPS
                ):

                    phase = "SEEK"

                    center_hold = 0

                    flyq.clear_motor_command()

            else:

                raise RuntimeError(
                    f"Unknown phase: "
                    f"{phase}"
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

            rows.append({
                "global_step": (
                    global_step
                ),

                "target_index": (
                    target_index
                ),

                "target_step": (
                    target_steps
                ),

                "phase": phase,

                "body_x": (
                    position[0]
                ),

                "body_y": (
                    position[1]
                ),

                "heading_deg": (
                    heading
                ),

                "target_x_world": (
                    target[0]
                ),

                "target_y_world": (
                    target[1]
                ),

                "initial_relative_bearing": (
                    initial_relative
                ),

                "initial_target_distance": (
                    initial_distance
                ),

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
            })

            should_print = (
                phase
                != previous_phase

                or target_steps % 25
                == 0

                or distance_after
                <= 0.20
            )

            if should_print:

                x_text = (
                    "OFF"
                    if target_x is None
                    else
                    f"{target_x:.3f}"
                )

                print(
                    f"{target_steps:>3} "
                    f"| global="
                    f"{global_step:>4} "
                    f"| phase="
                    f"{phase:>7} "
                    f"| x="
                    f"{x_text:>5} "
                    f"| turn="
                    f"{turn_action:>6} "
                    f"| move="
                    f"{movement_state:>4} "
                    f"| pos=("
                    f"{position[0]:+.3f},"
                    f"{position[1]:+.3f}) "
                    f"| head="
                    f"{heading:+6.1f}° "
                    f"| dist="
                    f"{distance_after:.3f}"
                )

            previous_phase = phase

            #
            # Evaluation only.
            #
            if (
                distance_after
                <= TARGET_RADIUS
            ):

                reached = True

                successes += 1

                print()
                print(
                    f"TARGET {target_index} "
                    f"REACHED ✅"
                )

                print(
                    f"Target steps : "
                    f"{target_steps}"
                )

                print(
                    f"Position     : "
                    f"({position[0]:+.3f}, "
                    f"{position[1]:+.3f})"
                )

                print(
                    f"Heading      : "
                    f"{heading:+.1f}°"
                )

                print(
                    f"Distance     : "
                    f"{distance_after:.3f}"
                )

                break

        (
            final_distance,
            final_relative,
        ) = geometry(
            position,
            heading,
            target,
        )

        if not reached:

            print()
            print(
                f"TARGET {target_index} "
                f"NOT REACHED ❌"
            )

            print(
                f"Final distance : "
                f"{final_distance:.3f}"
            )

            print(
                f"Minimum distance: "
                f"{min_distance:.3f}"
            )

        summaries.append({
            "target": target_index,

            "success": reached,

            "steps": target_steps,

            "relative_bearing": (
                initial_relative
            ),

            "initial_distance": (
                initial_distance
            ),

            "final_distance": (
                final_distance
            ),

            "min_distance": (
                min_distance
            ),

            "start_x": (
                initial_position[0]
            ),

            "start_y": (
                initial_position[1]
            ),

            "start_heading": (
                initial_heading
            ),

            "final_x": (
                position[0]
            ),

            "final_y": (
                position[1]
            ),

            "final_heading": (
                heading
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
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        "================================"
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "================================"
    )

    print()

    for result in summaries:

        status = (
            "✅"
            if result["success"]
            else "❌"
        )

        print(
            f"Target "
            f"{result['target']:>2} "
            f"| bearing="
            f"{result['relative_bearing']:+6.1f}° "
            f"| start_dist="
            f"{result['initial_distance']:.3f} "
            f"| steps="
            f"{result['steps']:>3} "
            f"| min_dist="
            f"{result['min_distance']:.3f} "
            f"| {status}"
        )

    print()

    print(
        f"Targets reached: "
        f"{successes}/"
        f"{N_TARGETS}"
    )

    print(
        f"Total neural steps: "
        f"{global_step}"
    )

    print(
        f"Total body path: "
        f"{total_path_length:.3f}"
    )

    print()

    print(
        "MaleCNS instances created: 1"
    )

    print(
        "Neural resets between targets: 0"
    )

    print()

    print(
        f"Telemetry saved → "
        f"{OUTPUT}"
    )

    del flyq
    gc.collect()


if __name__ == "__main__":
    main()
