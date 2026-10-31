import csv
import gc
import math
from pathlib import Path

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


# ============================================================
# Experiment
# ============================================================

TRIALS = [
    (+35.0, 0.80, +1),
    (-35.0, 0.80, -1),
    (+20.0, 0.75, -1),
    (-20.0, 0.75, +1),
    (+45.0, 0.85, +1),
    (-45.0, 0.85, -1),
]

TARGET_RADIUS = 0.12


# ============================================================
# Moving target
# ============================================================

#
# Target oscillates +/- this distance perpendicular
# to its original line of sight.
#
MOTION_AMPLITUDE = 0.15

#
# One full left-right-left oscillation.
#
MOTION_PERIOD_STEPS = 220


# ============================================================
# Body
# ============================================================

TURN_STEP_DEG = 2.0
MOVE_STEP = 0.01


# ============================================================
# Frozen V2 controller
# ============================================================

CENTER_HOLD_STEPS = 5

DRIVE_MOVE_STEPS = 6
DRIVE_MAX_NEURAL_STEPS = 20

RECOVERY_STEPS = 10


# ============================================================
# Limits
# ============================================================

MAX_STEPS_PER_TARGET = 850

OUTPUT = Path(
    "telemetry/moving_targets_2d.csv"
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
        relative_bearing,
    )


def create_anchor(
    position,
    heading_deg,
    relative_bearing,
    distance,
):
    absolute_bearing = wrap_angle(
        heading_deg
        + relative_bearing
    )

    angle = math.radians(
        absolute_bearing
    )

    anchor = (
        position[0]
        + distance
        * math.cos(angle),

        position[1]
        + distance
        * math.sin(angle),
    )

    #
    # Unit vector perpendicular to the
    # initial target direction.
    #
    tangent = (
        -math.sin(angle),
        math.cos(angle),
    )

    return (
        anchor,
        tangent,
    )


def moving_target_position(
    anchor,
    tangent,
    step,
    motion_direction,
):
    phase = (
        2.0
        * math.pi
        * step
        / MOTION_PERIOD_STEPS
    )

    offset = (
        motion_direction
        * MOTION_AMPLITUDE
        * math.sin(
            phase
        )
    )

    return (
        anchor[0]
        + tangent[0]
        * offset,

        anchor[1]
        + tangent[1]
        * offset,
    )


def main():

    print()
    print(
        "FlyQ Moving-Target Navigation"
    )

    print(
        "============================="
    )

    print()

    print(
        "6 moving targets"
    )

    print(
        "ONE persistent MaleCNS brain"
    )

    print(
        "NO neural reset between targets"
    )

    print(
        "Target moves during ALL phases"
    )

    print()

    print(
        f"Motion amplitude : "
        f"+/-{MOTION_AMPLITUDE:.2f}"
    )

    print(
        f"Motion period    : "
        f"{MOTION_PERIOD_STEPS} "
        f"neural steps"
    )

    print()

    # ========================================================
    # ONE brain for entire experiment
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

    # ========================================================
    # Trials
    # ========================================================

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
            f"MOVING TARGET "
            f"{trial_index}/"
            f"{len(TRIALS)}"
        )

        print(
            "================================"
        )

        print(
            f"Body position : "
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

        print(
            f"Anchor        : "
            f"({anchor[0]:+.3f}, "
            f"{anchor[1]:+.3f})"
        )

        print(
            "Motion        : "
            + (
                "A"
                if motion_direction > 0
                else "B"
            )
        )

        print()

        #
        # New target behavior state only.
        # MaleCNS remains untouched.
        #
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

        previous_phase = None

        reached = False

        min_distance = float(
            "inf"
        )

        target_steps = 0

        for _ in range(
            MAX_STEPS_PER_TARGET
        ):

            global_step += 1
            target_steps += 1

            #
            # TARGET MOVES EVERY NEURAL STEP.
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

                (
                    frame,
                    target_x,
                    visual_width,
                    visual_height,
                ) = render_target_depth(
                    relative_before,
                    distance_before,
                )

                neural = (
                    flyq.seek(
                        frame
                    )
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

                    total_path += math.hypot(
                        dx,
                        dy,
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

                (
                    frame,
                    target_x,
                    visual_width,
                    visual_height,
                ) = render_target_depth(
                    relative_before,
                    distance_before,
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

            #
            # Current target remains at its
            # current world position for this
            # neural step.
            #
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

                "trial_step": target_steps,

                "phase": phase,

                "body_x": position[0],
                "body_y": position[1],

                "heading_deg": heading,

                "target_x_world": (
                    target[0]
                ),

                "target_y_world": (
                    target[1]
                ),

                "target_anchor_x": (
                    anchor[0]
                ),

                "target_anchor_y": (
                    anchor[1]
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

                "turn_source": (
                    turn_source
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
                    f"| size="
                    f"{visual_width:>3}x"
                    f"{visual_height:<3} "
                    f"| turn="
                    f"{turn_action:>6} "
                    f"| move="
                    f"{movement_state:>4} "
                    f"| target=("
                    f"{target[0]:+.2f},"
                    f"{target[1]:+.2f}) "
                    f"| body=("
                    f"{position[0]:+.2f},"
                    f"{position[1]:+.2f}) "
                    f"| head="
                    f"{heading:+6.1f}° "
                    f"| dist="
                    f"{distance_after:.3f}"
                )

            previous_phase = phase

            # =================================================
            # Evaluation only
            # =================================================

            if (
                distance_after
                <= TARGET_RADIUS
            ):

                reached = True
                successes += 1

                print()
                print(
                    f"MOVING TARGET "
                    f"{trial_index} "
                    f"CAUGHT ✅"
                )

                print(
                    f"Steps    : "
                    f"{target_steps}"
                )

                print(
                    f"Body     : "
                    f"({position[0]:+.3f}, "
                    f"{position[1]:+.3f})"
                )

                print(
                    f"Target   : "
                    f"({target[0]:+.3f}, "
                    f"{target[1]:+.3f})"
                )

                print(
                    f"Distance : "
                    f"{distance_after:.3f}"
                )

                break

        #
        # Trial summary
        #
        final_target = (
            moving_target_position(
                anchor,
                tangent,
                target_steps,
                motion_direction,
            )
        )

        (
            final_distance,
            _,
        ) = geometry(
            position,
            heading,
            final_target,
        )

        if not reached:

            print()
            print(
                f"MOVING TARGET "
                f"{trial_index} "
                f"NOT CAUGHT ❌"
            )

            print(
                f"Minimum distance: "
                f"{min_distance:.3f}"
            )

        summaries.append({
            "trial": trial_index,
            "bearing": start_bearing,
            "start_distance": (
                start_distance
            ),
            "success": reached,
            "steps": target_steps,
            "min_distance": (
                min_distance
            ),
            "final_distance": (
                final_distance
            ),
        })

    # ========================================================
    # Telemetry
    # ========================================================

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

    # ========================================================
    # Final summary
    # ========================================================

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
            f"| start="
            f"{result['start_distance']:.2f} "
            f"| steps="
            f"{result['steps']:>3} "
            f"| min_dist="
            f"{result['min_distance']:.3f} "
            f"| {status}"
        )

    print()

    print(
        f"Moving targets caught: "
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
