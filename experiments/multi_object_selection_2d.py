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
)

from experiments.active_visual_search_2d import (
    SearchFlyQ,
)


# ============================================================
# Experiment
# ============================================================

TRIALS = [
    (+35.0, -35.0),
    (-35.0, +35.0),

    (+25.0, -40.0),
    (-25.0, +40.0),

    (+40.0, -20.0),
    (-40.0, +20.0),
]

OBJECT_DISTANCE = 0.80

TARGET_RADIUS = 0.12
DISTRACTOR_RADIUS = 0.12

MAX_STEPS_PER_TRIAL = 900

OUTPUT = Path(
    "telemetry/multi_object_selection_2d.csv"
)


# ============================================================
# Visual identity
# ============================================================

#
# Both objects are grayscale because our current
# R1-R6 bridge is luminance based.
#
# Target is deliberately much brighter.
#
TARGET_PIXEL = 255
DISTRACTOR_PIXEL = 110


# ============================================================
# Camera / perspective
# ============================================================

WIDTH = 320
HEIGHT = 240

FOV_DEG = 120.0
HALF_FOV_DEG = 60.0

REFERENCE_DISTANCE = 0.80

REFERENCE_WIDTH = 64
REFERENCE_HEIGHT = 160

MIN_SCALE = 0.65
MAX_SCALE = 1.50


# ============================================================
# Frozen navigation controller
# ============================================================

TURN_STEP_DEG = 2.0
MOVE_STEP = 0.01

CENTER_HOLD_STEPS = 5

DRIVE_MOVE_STEPS = 6
DRIVE_MAX_NEURAL_STEPS = 20

RECOVERY_STEPS = 10

RETINA_PRESENT_EPS = 1e-6


def make_object(
    position,
    heading_deg,
    relative_bearing,
):
    absolute_bearing = wrap_angle(
        heading_deg
        + relative_bearing
    )

    angle = math.radians(
        absolute_bearing
    )

    return (
        position[0]
        + OBJECT_DISTANCE
        * math.cos(angle),

        position[1]
        + OBJECT_DISTANCE
        * math.sin(angle),
    )


def geometry(
    position,
    heading_deg,
    point,
):
    dx = (
        point[0]
        - position[0]
    )

    dy = (
        point[1]
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

    relative = wrap_angle(
        absolute_bearing
        - heading_deg
    )

    return (
        distance,
        relative,
    )


def draw_object(
    frame,
    relative_bearing,
    distance,
    pixel,
):
    if (
        abs(relative_bearing)
        > HALF_FOV_DEG
    ):
        return (
            None,
            0,
            0,
        )

    x_fraction = (
        0.5
        - relative_bearing
        / FOV_DEG
    )

    safe_distance = max(
        float(distance),
        0.05,
    )

    scale = (
        REFERENCE_DISTANCE
        / safe_distance
    )

    scale = max(
        MIN_SCALE,
        min(
            MAX_SCALE,
            scale,
        ),
    )

    object_width = max(
        8,
        int(
            round(
                REFERENCE_WIDTH
                * scale
            )
        ),
    )

    object_height = max(
        20,
        int(
            round(
                REFERENCE_HEIGHT
                * scale
            )
        ),
    )

    object_width = min(
        object_width,
        WIDTH,
    )

    object_height = min(
        object_height,
        HEIGHT,
    )

    cx = int(
        x_fraction
        * (
            WIDTH - 1
        )
    )

    cy = HEIGHT // 2

    x1 = max(
        0,
        cx
        - object_width // 2,
    )

    x2 = min(
        WIDTH,
        cx
        + object_width // 2,
    )

    y1 = max(
        0,
        cy
        - object_height // 2,
    )

    y2 = min(
        HEIGHT,
        cy
        + object_height // 2,
    )

    #
    # Use maximum luminance if objects overlap.
    #
    region = frame[
        y1:y2,
        x1:x2,
    ]

    np.maximum(
        region,
        pixel,
        out=region,
    )

    return (
        x_fraction,
        object_width,
        object_height,
    )


def render_scene(
    position,
    heading,
    target,
    distractor,
):
    frame = np.zeros(
        (
            HEIGHT,
            WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    (
        target_distance,
        target_relative,
    ) = geometry(
        position,
        heading,
        target,
    )

    (
        distractor_distance,
        distractor_relative,
    ) = geometry(
        position,
        heading,
        distractor,
    )

    #
    # Draw dim distractor first.
    #
    (
        distractor_x,
        distractor_w,
        distractor_h,
    ) = draw_object(
        frame,
        distractor_relative,
        distractor_distance,
        DISTRACTOR_PIXEL,
    )

    #
    # Bright designated target wins overlap.
    #
    (
        target_x,
        target_w,
        target_h,
    ) = draw_object(
        frame,
        target_relative,
        target_distance,
        TARGET_PIXEL,
    )

    return {
        "frame": frame,

        "target_distance": (
            target_distance
        ),

        "target_relative": (
            target_relative
        ),

        "target_x": (
            target_x
        ),

        "target_w": (
            target_w
        ),

        "target_h": (
            target_h
        ),

        "distractor_distance": (
            distractor_distance
        ),

        "distractor_relative": (
            distractor_relative
        ),

        "distractor_x": (
            distractor_x
        ),

        "distractor_w": (
            distractor_w
        ),

        "distractor_h": (
            distractor_h
        ),
    }


def main():

    print()
    print(
        "FlyQ Multiple-Object Selection"
    )

    print(
        "=============================="
    )

    print()

    print(
        "Bright object = designated target"
    )

    print(
        "Dim object    = distractor"
    )

    print()

    print(
        "Controller receives only combined RGB."
    )

    print(
        "No target bearing is passed to controller."
    )

    print(
        "ONE persistent MaleCNS brain."
    )

    print()

    # ========================================================
    # ONE brain
    # ========================================================

    flyq = SearchFlyQ()

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
    total_path = 0.0

    successes = 0
    distractor_contacts = 0

    rows = []
    summaries = []

    # ========================================================
    # Trials
    # ========================================================

    for trial_index, (
        target_bearing,
        distractor_bearing,
    ) in enumerate(
        TRIALS,
        start=1,
    ):

        target = make_object(
            position,
            heading,
            target_bearing,
        )

        distractor = make_object(
            position,
            heading,
            distractor_bearing,
        )

        print()
        print(
            "================================"
        )

        print(
            f"SELECTION TRIAL "
            f"{trial_index}/"
            f"{len(TRIALS)}"
        )

        print(
            "================================"
        )

        print(
            f"Body heading       : "
            f"{heading:+.1f}°"
        )

        print(
            f"BRIGHT target      : "
            f"{target_bearing:+.1f}°"
        )

        print(
            f"DIM distractor     : "
            f"{distractor_bearing:+.1f}°"
        )

        print()

        #
        # Behavioral engineering state only.
        #
        flyq.reset_turn_filters()

        phase = (
            "SEEK"
            if trial_index == 1
            else "RECOVER"
        )

        movement_decoder = (
            MovementBurstDecoder()
        )

        center_hold = 0
        recovery_steps = 0

        drive_neural_steps = 0
        drive_move_steps = 0

        reached = False
        hit_distractor = False

        min_target_distance = float(
            "inf"
        )

        min_distractor_distance = float(
            "inf"
        )

        trial_steps = 0
        previous_phase = None

        for _ in range(
            MAX_STEPS_PER_TRIAL
        ):

            global_step += 1
            trial_steps += 1

            scene = render_scene(
                position,
                heading,
                target,
                distractor,
            )

            target_distance = (
                scene[
                    "target_distance"
                ]
            )

            distractor_distance = (
                scene[
                    "distractor_distance"
                ]
            )

            min_target_distance = min(
                min_target_distance,
                target_distance,
            )

            min_distractor_distance = min(
                min_distractor_distance,
                distractor_distance,
            )

            retina_mean = 0.0

            neural_direction = (
                "CENTER"
            )

            turn_action = (
                "CENTER"
            )

            filtered_turn = 0.0

            movement_hz = 0.0
            movement_state = "STOP"

            # ================================================
            # SEEK
            # ================================================

            if phase == "SEEK":

                neural = (
                    flyq.target_observation(
                        scene["frame"]
                    )
                )

                retina_mean = (
                    neural[
                        "retina_mean"
                    ]
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

                #
                # Never walk if the entire scene
                # has vanished.
                #
                if (
                    retina_mean
                    <= RETINA_PRESENT_EPS
                ):

                    center_hold = 0

                    turn_action = (
                        "CENTER"
                    )

                else:

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
                        and
                        turn_action
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

                movement_hz = (
                    flyq.movement_step()
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

                    flyq.reset_turn_filters()

            # ================================================
            # RECOVER
            # ================================================

            elif phase == "RECOVER":

                neural = (
                    flyq.target_observation(
                        scene["frame"]
                    )
                )

                retina_mean = (
                    neural[
                        "retina_mean"
                    ]
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

                    flyq.reset_turn_filters()

            else:

                raise RuntimeError(
                    phase
                )

            #
            # Recompute geometry after movement.
            #
            (
                target_distance_after,
                target_relative_after,
            ) = geometry(
                position,
                heading,
                target,
            )

            (
                distractor_distance_after,
                distractor_relative_after,
            ) = geometry(
                position,
                heading,
                distractor,
            )

            min_target_distance = min(
                min_target_distance,
                target_distance_after,
            )

            min_distractor_distance = min(
                min_distractor_distance,
                distractor_distance_after,
            )

            rows.append({
                "global_step": (
                    global_step
                ),

                "trial": (
                    trial_index
                ),

                "trial_step": (
                    trial_steps
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

                "distractor_x_world": (
                    distractor[0]
                ),

                "distractor_y_world": (
                    distractor[1]
                ),

                "target_distance": (
                    target_distance_after
                ),

                "distractor_distance": (
                    distractor_distance_after
                ),

                "target_relative": (
                    target_relative_after
                ),

                "distractor_relative": (
                    distractor_relative_after
                ),

                "target_screen_x": (
                    ""
                    if (
                        scene[
                            "target_x"
                        ]
                        is None
                    )
                    else
                    scene[
                        "target_x"
                    ]
                ),

                "distractor_screen_x": (
                    ""
                    if (
                        scene[
                            "distractor_x"
                        ]
                        is None
                    )
                    else
                    scene[
                        "distractor_x"
                    ]
                ),

                "retina_mean": (
                    retina_mean
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
                trial_steps % 25 == 0
                or
                target_distance_after
                <= 0.20
                or
                distractor_distance_after
                <= 0.20
            ):

                tx = (
                    "OFF"
                    if (
                        scene[
                            "target_x"
                        ]
                        is None
                    )
                    else
                    f"{scene['target_x']:.3f}"
                )

                dx_text = (
                    "OFF"
                    if (
                        scene[
                            "distractor_x"
                        ]
                        is None
                    )
                    else
                    f"{scene['distractor_x']:.3f}"
                )

                print(
                    f"{trial_steps:>3} "
                    f"| phase="
                    f"{phase:>7} "
                    f"| target_x="
                    f"{tx:>5} "
                    f"| distractor_x="
                    f"{dx_text:>5} "
                    f"| turn="
                    f"{turn_action:>6} "
                    f"| target_dist="
                    f"{target_distance_after:.3f} "
                    f"| distractor_dist="
                    f"{distractor_distance_after:.3f}"
                )

            previous_phase = phase

            # ================================================
            # Evaluation only
            # ================================================

            if (
                target_distance_after
                <= TARGET_RADIUS
            ):

                reached = True
                successes += 1

                print()
                print(
                    f"BRIGHT TARGET "
                    f"{trial_index} "
                    f"REACHED ✅"
                )

                print(
                    f"Steps           : "
                    f"{trial_steps}"
                )

                print(
                    f"Target distance : "
                    f"{target_distance_after:.3f}"
                )

                print(
                    f"Distractor dist : "
                    f"{distractor_distance_after:.3f}"
                )

                break

            if (
                distractor_distance_after
                <= DISTRACTOR_RADIUS
            ):

                hit_distractor = True

                distractor_contacts += 1

                print()
                print(
                    f"DIM DISTRACTOR "
                    f"{trial_index} "
                    f"CONTACTED ❌"
                )

                print(
                    f"Steps           : "
                    f"{trial_steps}"
                )

                print(
                    f"Target distance : "
                    f"{target_distance_after:.3f}"
                )

                print(
                    f"Distractor dist : "
                    f"{distractor_distance_after:.3f}"
                )

                break

        if (
            not reached
            and
            not hit_distractor
        ):

            print()
            print(
                f"TRIAL {trial_index} "
                f"TIMED OUT ❌"
            )

        summaries.append({
            "trial": (
                trial_index
            ),

            "target_bearing": (
                target_bearing
            ),

            "distractor_bearing": (
                distractor_bearing
            ),

            "success": reached,

            "hit_distractor": (
                hit_distractor
            ),

            "steps": (
                trial_steps
            ),

            "min_target_distance": (
                min_target_distance
            ),

            "min_distractor_distance": (
                min_distractor_distance
            ),
        })

    # ========================================================
    # Save telemetry
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
    # Summary
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

        if result["success"]:

            status = "TARGET ✅"

        elif result[
            "hit_distractor"
        ]:

            status = (
                "DISTRACTOR ❌"
            )

        else:

            status = (
                "TIMEOUT ❌"
            )

        print(
            f"Trial "
            f"{result['trial']} "
            f"| target="
            f"{result['target_bearing']:+6.1f}° "
            f"| distractor="
            f"{result['distractor_bearing']:+6.1f}° "
            f"| steps="
            f"{result['steps']:>3} "
            f"| min_target="
            f"{result['min_target_distance']:.3f} "
            f"| min_distractor="
            f"{result['min_distractor_distance']:.3f} "
            f"| {status}"
        )

    print()

    print(
        f"Bright targets reached: "
        f"{successes}/"
        f"{len(TRIALS)}"
    )

    print(
        f"Distractor contacts: "
        f"{distractor_contacts}"
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
        "Neural resets between trials: 0"
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
