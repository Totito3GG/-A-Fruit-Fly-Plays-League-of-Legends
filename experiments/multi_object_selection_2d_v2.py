import csv
import gc
import math
from pathlib import Path

import numpy as np

from experiments.pixel_target_centering import wrap_angle
from experiments.pixel_navigation_2d_v2 import MovementBurstDecoder
from experiments.active_visual_search_2d import SearchFlyQ

from experiments.multi_object_selection_2d import (
    TRIALS,
    TARGET_RADIUS,
    DISTRACTOR_RADIUS,
    render_scene,
    geometry,
    make_object,
)


OUTPUT = Path(
    "telemetry/multi_object_selection_2d_v2.csv"
)

MAX_STEPS_PER_TRIAL = 1000


# ============================================================
# Frozen body / navigation
# ============================================================

TURN_STEP_DEG = 2.0
MOVE_STEP = 0.01

CENTER_HOLD_STEPS = 5

DRIVE_MOVE_STEPS = 6
DRIVE_MAX_NEURAL_STEPS = 20

RECOVERY_STEPS = 10


# ============================================================
# Engineered visual salience gate
# ============================================================

#
# V1 scene:
#
#   bright target    = 255
#   dim distractor   = 110
#
# Gate is intentionally between them.
#
# IMPORTANT:
# This is an engineered RGB salience selector.
# It does NOT use target coordinates/bearing/distance.
#
SALIENCE_THRESHOLD = 200


def salience_gate(frame):
    """
    Combined RGB -> bright-stimulus-only RGB.

    No world geometry is available here.
    """

    intensity = np.max(
        frame,
        axis=2,
    )

    mask = (
        intensity
        >= SALIENCE_THRESHOLD
    )

    output = np.zeros_like(
        frame
    )

    output[mask] = frame[mask]

    return output


# ============================================================
# Lost-target / search
# ============================================================

RETINA_PRESENT_EPS = 1e-6

LOSS_CONFIRM_STEPS = 5
FOUND_CONFIRM_STEPS = 5

SEARCH_LIMIT_DEG = 80.0


def main():

    print()
    print(
        "FlyQ Multiple-Object Selection V2"
    )
    print(
        "================================="
    )
    print()

    print(
        "Combined RGB scene contains:"
    )
    print(
        "  bright target"
    )
    print(
        "  dim distractor"
    )
    print()

    print(
        "Engineered RGB salience gate:"
    )
    print(
        f"  threshold >= {SALIENCE_THRESHOLD}"
    )
    print()

    print(
        "Gate receives NO target coordinates."
    )
    print(
        "MaleCNS receives gated RGB."
    )
    print(
        "DNp20/DNpe017 controller remains frozen."
    )
    print(
        "ONE persistent MaleCNS brain."
    )
    print()

    # ========================================================
    # One persistent MaleCNS
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

    total_search_events = 0

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
            f"V2 SELECTION TRIAL "
            f"{trial_index}/"
            f"{len(TRIALS)}"
        )
        print(
            "================================"
        )

        print(
            f"Body heading   : "
            f"{heading:+.1f}°"
        )

        print(
            f"BRIGHT target  : "
            f"{target_bearing:+.1f}°"
        )

        print(
            f"DIM distractor : "
            f"{distractor_bearing:+.1f}°"
        )

        print()

        #
        # Engineering state only.
        # MaleCNS itself is NOT reset.
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

        loss_count = 0

        search_direction = "LEFT"
        search_offset = 0.0
        found_count = 0

        search_events = 0
        search_steps = 0

        reached = False
        hit_distractor = False

        min_target_distance = float(
            "inf"
        )

        min_distractor_distance = float(
            "inf"
        )

        previous_phase = None

        trial_steps = 0

        # ====================================================
        # Trial loop
        # ====================================================

        for _ in range(
            MAX_STEPS_PER_TRIAL
        ):

            trial_steps += 1
            global_step += 1

            # ================================================
            # WORLD / RENDERER
            # ================================================

            scene = render_scene(
                position,
                heading,
                target,
                distractor,
            )

            #
            # This is the ONLY visual selector.
            #
            # Input:
            #     RGB pixels
            #
            # It has no geometry.
            #
            controller_frame = (
                salience_gate(
                    scene["frame"]
                )
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

            movement_state = (
                "STOP"
            )

            # ================================================
            # SEEK
            # ================================================

            if phase == "SEEK":

                neural = (
                    flyq.target_observation(
                        controller_frame
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

                visible = (
                    retina_mean
                    > RETINA_PRESENT_EPS
                )

                # ============================================
                # Bright target disappeared
                # ============================================

                if not visible:

                    loss_count += 1

                    center_hold = 0

                    turn_action = (
                        "CENTER"
                    )

                    if (
                        loss_count
                        >= LOSS_CONFIRM_STEPS
                    ):

                        phase = "SEARCH"

                        search_events += 1
                        total_search_events += 1

                        search_direction = (
                            "LEFT"
                        )

                        search_offset = 0.0

                        found_count = 0

                        flyq.reset_turn_filters()

                        print()
                        print(
                            f">>> BRIGHT TARGET LOST — "
                            f"SEARCH START "
                            f"at step {trial_steps}"
                        )

                # ============================================
                # Bright target visible
                # ============================================

                else:

                    loss_count = 0

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
            # SEARCH
            # ================================================

            elif phase == "SEARCH":

                search_steps += 1

                neural = (
                    flyq.search_observation(
                        controller_frame
                    )
                )

                retina_mean = (
                    neural[
                        "retina_mean"
                    ]
                )

                movement_hz = (
                    neural[
                        "movement_hz"
                    ]
                )

                visible = (
                    retina_mean
                    > RETINA_PRESENT_EPS
                )

                #
                # Stop scan once the bright target
                # creates retinal activity again.
                #
                if visible:

                    found_count += 1

                    turn_action = (
                        "CENTER"
                    )

                    if (
                        found_count
                        >= FOUND_CONFIRM_STEPS
                    ):

                        phase = "SEEK"

                        loss_count = 0
                        center_hold = 0

                        flyq.reset_turn_filters()

                        print()
                        print(
                            f">>> BRIGHT TARGET "
                            f"REACQUIRED "
                            f"at step "
                            f"{trial_steps}"
                        )

                else:

                    found_count = 0

                    #
                    # ENGINEERED SCAN.
                    #
                    # Critically:
                    #
                    # search direction is determined
                    # only by internal sweep state.
                    #
                    # No target coordinates.
                    #
                    if (
                        search_direction
                        == "LEFT"
                    ):

                        heading = wrap_angle(
                            heading
                            + TURN_STEP_DEG
                        )

                        search_offset += (
                            TURN_STEP_DEG
                        )

                        turn_action = (
                            "SCAN_LEFT"
                        )

                        if (
                            search_offset
                            >= SEARCH_LIMIT_DEG
                        ):

                            search_direction = (
                                "RIGHT"
                            )

                    else:

                        heading = wrap_angle(
                            heading
                            - TURN_STEP_DEG
                        )

                        search_offset -= (
                            TURN_STEP_DEG
                        )

                        turn_action = (
                            "SCAN_RIGHT"
                        )

                        if (
                            search_offset
                            <= -SEARCH_LIMIT_DEG
                        ):

                            search_direction = (
                                "LEFT"
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
                    loss_count = 0

                    flyq.reset_turn_filters()

            # ================================================
            # RECOVER
            # ================================================

            elif phase == "RECOVER":

                neural = (
                    flyq.target_observation(
                        controller_frame
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

                visible = (
                    retina_mean
                    > RETINA_PRESENT_EPS
                )

                if not visible:

                    loss_count += 1

                    if (
                        loss_count
                        >= LOSS_CONFIRM_STEPS
                    ):

                        phase = "SEARCH"

                        search_events += 1
                        total_search_events += 1

                        search_direction = (
                            "LEFT"
                        )

                        search_offset = 0.0

                        found_count = 0

                        flyq.reset_turn_filters()

                        print()
                        print(
                            f">>> BRIGHT TARGET LOST "
                            f"during RECOVER — "
                            f"SEARCH at step "
                            f"{trial_steps}"
                        )

                else:

                    loss_count = 0

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

            # ================================================
            # WORLD / EVALUATION ONLY
            # ================================================

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

            min_target_distance = min(
                min_target_distance,
                target_distance,
            )

            min_distractor_distance = min(
                min_distractor_distance,
                distractor_distance,
            )

            raw_pixels = int(
                np.count_nonzero(
                    scene["frame"]
                )
            )

            gated_pixels = int(
                np.count_nonzero(
                    controller_frame
                )
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

                "target_distance": (
                    target_distance
                ),

                "distractor_distance": (
                    distractor_distance
                ),

                "target_relative": (
                    target_relative
                ),

                "distractor_relative": (
                    distractor_relative
                ),

                "raw_nonzero_pixels": (
                    raw_pixels
                ),

                "gated_nonzero_pixels": (
                    gated_pixels
                ),

                "retina_mean": (
                    retina_mean
                ),

                "neural_direction": (
                    neural_direction
                ),

                "turn_action": (
                    turn_action
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

                "search_events": (
                    search_events
                ),

                "search_offset_deg": (
                    search_offset
                ),
            })

            # ================================================
            # Console telemetry
            # ================================================

            if (
                phase
                != previous_phase
                or
                trial_steps % 25 == 0
                or
                target_distance <= 0.20
                or
                distractor_distance <= 0.20
            ):

                print(
                    f"{trial_steps:>3} "
                    f"| phase="
                    f"{phase:>7} "
                    f"| retina="
                    f"{retina_mean:.5f} "
                    f"| turn="
                    f"{turn_action:>10} "
                    f"| target="
                    f"{target_distance:.3f} "
                    f"| distractor="
                    f"{distractor_distance:.3f} "
                    f"| gate_px="
                    f"{gated_pixels}"
                )

            previous_phase = phase

            # ================================================
            # Target success
            # ================================================

            if (
                target_distance
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
                    f"{target_distance:.3f}"
                )

                print(
                    f"Distractor dist : "
                    f"{distractor_distance:.3f}"
                )

                print(
                    f"Search events   : "
                    f"{search_events}"
                )

                break

            # ================================================
            # Wrong-object contact
            # ================================================

            if (
                distractor_distance
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
                    f"Target distance : "
                    f"{target_distance:.3f}"
                )

                print(
                    f"Distractor dist : "
                    f"{distractor_distance:.3f}"
                )

                break

        if (
            not reached
            and
            not hit_distractor
        ):

            print()
            print(
                f"V2 TRIAL "
                f"{trial_index} "
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

            "success": (
                reached
            ),

            "hit_distractor": (
                hit_distractor
            ),

            "steps": (
                trial_steps
            ),

            "search_events": (
                search_events
            ),

            "search_steps": (
                search_steps
            ),

            "min_target": (
                min_target_distance
            ),

            "min_distractor": (
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
    # Final summary
    # ========================================================

    print()
    print(
        "================================"
    )
    print(
        "FINAL SUMMARY — V2"
    )
    print(
        "================================"
    )
    print()

    for result in summaries:

        if result["success"]:

            status = (
                "TARGET ✅"
            )

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
            f"{result['steps']:>4} "
            f"| search="
            f"{result['search_events']} "
            f"| min_target="
            f"{result['min_target']:.3f} "
            f"| min_distractor="
            f"{result['min_distractor']:.3f} "
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
        f"Total search events: "
        f"{total_search_events}"
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
