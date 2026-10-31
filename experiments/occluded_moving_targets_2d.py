import csv
import gc
import math
from pathlib import Path

import numpy as np

from experiments.pixel_target_centering import (
    wrap_angle,
)

from experiments.pixel_navigation_2d_v2 import (
    MovementBurstDecoder,
    movement_cue_frame,
)

from experiments.sequential_depth_targets_2d import (
    PersistentFlyQ,
    render_target_depth,
)

from experiments.moving_targets_2d import (
    TRIALS,
    TARGET_RADIUS,
    create_anchor,
    geometry,
    moving_target_position,
)


# ============================================================
# Frozen controller
# ============================================================

TURN_STEP_DEG = 2.0
MOVE_STEP = 0.01

CENTER_HOLD_STEPS = 5

DRIVE_MOVE_STEPS = 6
DRIVE_MAX_NEURAL_STEPS = 20

RECOVERY_STEPS = 10


# ============================================================
# Occlusion
# ============================================================

#
# Target disappears for 30 neural steps
# every 150 steps.
#
# It CONTINUES MOVING while invisible.
#
OCCLUSION_START_STEP = 75
OCCLUSION_PERIOD_STEPS = 150
OCCLUSION_DURATION_STEPS = 30


# ============================================================
# Limits
# ============================================================

MAX_STEPS_PER_TARGET = 950

OUTPUT = Path(
    "telemetry/occluded_moving_targets_2d.csv"
)


def is_occluded(step):
    if step < OCCLUSION_START_STEP:
        return False

    phase = (
        step
        - OCCLUSION_START_STEP
    ) % OCCLUSION_PERIOD_STEPS

    return (
        phase
        < OCCLUSION_DURATION_STEPS
    )


def black_frame():
    return np.zeros(
        (240, 320, 3),
        dtype=np.uint8,
    )


def main():

    print()
    print(
        "FlyQ Occluded Moving-Target Navigation"
    )

    print(
        "======================================"
    )

    print()
    print(
        "Target continues moving while invisible."
    )

    print(
        "Controller receives NO target coordinates."
    )

    print(
        "NO special search behavior is added."
    )

    print()

    print(
        f"Occlusion starts : step "
        f"{OCCLUSION_START_STEP}"
    )

    print(
        f"Hidden duration  : "
        f"{OCCLUSION_DURATION_STEPS} steps"
    )

    print(
        f"Occlusion period : "
        f"{OCCLUSION_PERIOD_STEPS} steps"
    )

    print()

    # ========================================================
    # ONE persistent brain
    # ========================================================

    flyq = PersistentFlyQ()

    baseline = (
        flyq.calibrate_black()
    )

    print(
        f"Black DNp20 baseline: "
        f"{baseline:+.2f} Hz"
    )

    position = [
        0.0,
        0.0,
    ]

    heading = 0.0

    global_step = 0

    successes = 0

    total_path = 0.0

    rows = []
    summaries = []

    for trial_index, (
        start_bearing,
        start_distance,
        motion_direction,
    ) in enumerate(
        TRIALS,
        start=1,
    ):

        (
            anchor,
            tangent,
        ) = create_anchor(
            position,
            heading,
            start_bearing,
            start_distance,
        )

        print()
        print(
            "================================"
        )

        print(
            f"OCCLUDED TARGET "
            f"{trial_index}/{len(TRIALS)}"
        )

        print(
            "================================"
        )

        print(
            f"Body          : "
            f"({position[0]:+.3f}, "
            f"{position[1]:+.3f})"
        )

        print(
            f"Heading       : "
            f"{heading:+.1f}°"
        )

        print(
            f"Start bearing : "
            f"{start_bearing:+.1f}°"
        )

        print(
            f"Start distance: "
            f"{start_distance:.3f}"
        )

        print()

        if trial_index == 1:
            phase = "SEEK"
        else:
            phase = "RECOVER"

        flyq.clear_motor_command()

        movement_decoder = (
            MovementBurstDecoder()
        )

        center_hold = 0
        recovery_steps = 0

        drive_neural_steps = 0
        drive_move_steps = 0

        target_steps = 0

        reached = False

        min_distance = float(
            "inf"
        )

        occluded_steps = 0
        occlusion_events = 0

        previous_occluded = False
        previous_phase = None

        for _ in range(
            MAX_STEPS_PER_TARGET
        ):

            global_step += 1
            target_steps += 1

            #
            # Target continues moving regardless
            # of whether FlyQ can see it.
            #
            target = (
                moving_target_position(
                    anchor,
                    tangent,
                    target_steps,
                    motion_direction,
                )
            )

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

            occluded = (
                is_occluded(
                    target_steps
                )
            )

            if occluded:
                occluded_steps += 1

            if (
                occluded
                and not previous_occluded
            ):
                occlusion_events += 1

                print()
                print(
                    f">>> TARGET OCCLUDED "
                    f"at step {target_steps}"
                )

            if (
                not occluded
                and previous_occluded
            ):
                print(
                    f">>> TARGET VISIBLE AGAIN "
                    f"at step {target_steps}"
                )
                print()

            previous_occluded = (
                occluded
            )

            target_x = None

            visual_width = 0
            visual_height = 0

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

            # =================================================
            # SEEK
            # =================================================

            if phase == "SEEK":

                if occluded:

                    frame = (
                        black_frame()
                    )

                else:

                    (
                        frame,
                        target_x,
                        visual_width,
                        visual_height,
                    ) = (
                        render_target_depth(
                            relative_before,
                            distance_before,
                        )
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

            # =================================================
            # DRIVE
            # =================================================

            elif phase == "DRIVE":

                #
                # Same engineered DNpe017 cue.
                #
                frame = (
                    movement_cue_frame()
                )

                movement_hz = (
                    flyq.movement_step(
                        frame
                    )
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

                    angle = math.radians(
                        heading
                    )

                    dx = (
                        MOVE_STEP
                        * math.cos(
                            angle
                        )
                    )

                    dy = (
                        MOVE_STEP
                        * math.sin(
                            angle
                        )
                    )

                    position[0] += dx
                    position[1] += dy

                    total_path += (
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

            # =================================================
            # RECOVER
            # =================================================

            elif phase == "RECOVER":

                if occluded:

                    frame = (
                        black_frame()
                    )

                else:

                    (
                        frame,
                        target_x,
                        visual_width,
                        visual_height,
                    ) = (
                        render_target_depth(
                            relative_before,
                            distance_before,
                        )
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
                    phase
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
                "global_step": global_step,

                "trial": trial_index,

                "trial_step": (
                    target_steps
                ),

                "phase": phase,

                "occluded": int(
                    occluded
                ),

                "body_x": position[0],
                "body_y": position[1],

                "heading_deg": heading,

                "target_x_world": (
                    target[0]
                ),

                "target_y_world": (
                    target[1]
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

                "visual_target_width": (
                    visual_width
                ),

                "visual_target_height": (
                    visual_height
                ),

                "turn_action": (
                    turn_action
                ),

                "neural_direction": (
                    neural_direction
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

            if (
                phase != previous_phase
                or
                target_steps % 25 == 0
                or
                distance_after <= 0.20
            ):

                visibility = (
                    "HIDDEN"
                    if occluded
                    else "VISIBLE"
                )

                x_text = (
                    "OFF"
                    if target_x is None
                    else
                    f"{target_x:.3f}"
                )

                print(
                    f"{target_steps:>3} "
                    f"| phase="
                    f"{phase:>7} "
                    f"| {visibility:>7} "
                    f"| x="
                    f"{x_text:>5} "
                    f"| turn="
                    f"{turn_action:>6} "
                    f"| move="
                    f"{movement_state:>4} "
                    f"| body=("
                    f"{position[0]:+.2f},"
                    f"{position[1]:+.2f}) "
                    f"| target=("
                    f"{target[0]:+.2f},"
                    f"{target[1]:+.2f}) "
                    f"| dist="
                    f"{distance_after:.3f}"
                )

            previous_phase = phase

            #
            # World/evaluation only.
            #
            if (
                distance_after
                <= TARGET_RADIUS
            ):

                reached = True
                successes += 1

                print()
                print(
                    f"OCCLUDED MOVING TARGET "
                    f"{trial_index} "
                    f"CAUGHT ✅"
                )

                print(
                    f"Steps            : "
                    f"{target_steps}"
                )

                print(
                    f"Occlusion events : "
                    f"{occlusion_events}"
                )

                print(
                    f"Hidden steps     : "
                    f"{occluded_steps}"
                )

                print(
                    f"Distance         : "
                    f"{distance_after:.3f}"
                )

                break

        if not reached:

            print()
            print(
                f"OCCLUDED MOVING TARGET "
                f"{trial_index} "
                f"NOT CAUGHT ❌"
            )

            print(
                f"Minimum distance : "
                f"{min_distance:.3f}"
            )

        summaries.append({
            "trial": trial_index,
            "bearing": start_bearing,
            "success": reached,
            "steps": target_steps,
            "min_distance": (
                min_distance
            ),
            "occlusions": (
                occlusion_events
            ),
            "hidden_steps": (
                occluded_steps
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
            if result[
                "success"
            ]
            else "❌"
        )

        print(
            f"Trial "
            f"{result['trial']} "
            f"| bearing="
            f"{result['bearing']:+6.1f}° "
            f"| steps="
            f"{result['steps']:>3} "
            f"| hidden="
            f"{result['hidden_steps']:>3} "
            f"| occlusions="
            f"{result['occlusions']} "
            f"| min_dist="
            f"{result['min_distance']:.3f} "
            f"| {status}"
        )

    print()

    print(
        f"Occluded targets caught: "
        f"{successes}/"
        f"{len(TRIALS)}"
    )

    print(
        f"Total neural steps: "
        f"{global_step}"
    )

    print(
        f"Total body path: "
        f"{total_path:.3f}"
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
