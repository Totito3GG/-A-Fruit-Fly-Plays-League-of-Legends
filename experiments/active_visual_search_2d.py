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

from experiments.pixel_navigation_2d_v2 import (
    MovementBurstDecoder,
    movement_cue_frame,
)

from experiments.sequential_depth_targets_2d import (
    render_target_depth,
)


# ============================================================
# Experiment
# ============================================================

TARGET_BEARINGS = [
    +80.0,
    -80.0,
    +100.0,
    -100.0,
    +120.0,
    -120.0,
]

TARGET_DISTANCE = 0.80
TARGET_RADIUS = 0.12

MAX_STEPS_PER_TARGET = 1000

OUTPUT = Path(
    "telemetry/active_visual_search_2d.csv"
)


# ============================================================
# Body
# ============================================================

TURN_STEP_DEG = 2.0
MOVE_STEP = 0.01


# ============================================================
# Frozen navigation policy
# ============================================================

CENTER_HOLD_STEPS = 5

DRIVE_MOVE_STEPS = 6
DRIVE_MAX_NEURAL_STEPS = 20

RECOVERY_STEPS = 10


# ============================================================
# Search
# ============================================================

#
# Black stimulus produces exactly zero retinal
# luminance in the current DoomFly projection.
#
# We deliberately use the neural runtime's
# retinal measurement rather than target_x,
# relative bearing or world coordinates.
#
RETINA_PRESENT_EPS = 1e-6

#
# Number of consecutive target-observation
# frames with no retinal signal before search.
#
LOSS_CONFIRM_STEPS = 5

#
# Require the visual target to remain detectable
# for several frames before handing control back
# to DNp20.
#
FOUND_CONFIRM_STEPS = 5

#
# Search sweep relative to the heading where
# search began.
#
# +80° <-> -80°
#
SEARCH_LIMIT_DEG = 80.0


# ============================================================
# Calibration
# ============================================================

WARMUP_STEPS = 30


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


def make_target(
    position,
    heading_deg,
    relative_bearing,
):
    """
    WORLD/RENDERER ONLY.

    Controller never receives these coordinates.
    """

    absolute_bearing = wrap_angle(
        heading_deg
        + relative_bearing
    )

    angle = math.radians(
        absolute_bearing
    )

    return (
        position[0]
        + TARGET_DISTANCE
        * math.cos(angle),

        position[1]
        + TARGET_DISTANCE
        * math.sin(angle),
    )


class SearchFlyQ:
    """
    ONE persistent MaleCNS brain.

    Engineering decoders may be cleared.
    NativeBrain itself is never recreated/reset.
    """

    def __init__(self):

        self.runtime = (
            FlyQBrainRuntime()
        )

        self.decoder = (
            PixelTurnDecoder()
        )

        self.actuator = (
            PersistentTurnActuator()
        )

        self.baseline = 0.0

    def reset_turn_filters(self):
        """
        Engineering state only.

        Does NOT reset MaleCNS.
        """

        self.decoder = (
            PixelTurnDecoder()
        )

        self.actuator = (
            PersistentTurnActuator()
        )

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

        self.baseline = float(
            np.mean(
                values[-10:]
            )
        )

        self.reset_turn_filters()

        return self.baseline

    def target_observation(
        self,
        frame,
    ):
        """
        Target RGB -> MaleCNS -> retinal activity
                   -> DNp20 engineering decoder.
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

        retina_mean = float(
            result[
                "retina_mean"
            ]
        )

        movement_hz = float(
            result[
                "movement_raw_hz"
            ]
        )

        delta = (
            raw_turn
            - self.baseline
        )

        (
            direction,
            filtered,
            ready,
        ) = (
            self.decoder
            .update(
                delta
            )
        )

        (
            action,
            source,
        ) = (
            self.actuator
            .update(
                direction
            )
        )

        return {
            "retina_mean": (
                retina_mean
            ),

            "direction": direction,

            "action": action,

            "source": source,

            "filtered": filtered,

            "ready": ready,

            "movement_hz": (
                movement_hz
            ),
        }

    def search_observation(
        self,
        frame,
    ):
        """
        During engineered search we only ask:
            does MaleCNS retinal input detect light?

        We deliberately do NOT use world geometry
        to decide search direction.
        """

        result = (
            self.runtime
            .process_frame(
                frame
            )
        )

        return {
            "retina_mean": float(
                result[
                    "retina_mean"
                ]
            ),

            "movement_hz": float(
                result[
                    "movement_raw_hz"
                ]
            ),
        }

    def movement_step(
        self,
    ):
        result = (
            self.runtime
            .process_frame(
                movement_cue_frame()
            )
        )

        return float(
            result[
                "movement_raw_hz"
            ]
        )


def main():

    print()
    print(
        "FlyQ Active Visual Search"
    )

    print(
        "========================="
    )

    print()

    print(
        "Targets START outside the camera FOV."
    )

    print(
        "Search trigger = retinal signal loss."
    )

    print(
        "Search direction uses NO target position."
    )

    print(
        "ONE persistent MaleCNS brain."
    )

    print()

    flyq = SearchFlyQ()

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

    total_path = 0.0

    successes = 0

    total_search_events = 0

    rows = []
    summaries = []

    for trial_index, start_bearing in enumerate(
        TARGET_BEARINGS,
        start=1,
    ):

        target = make_target(
            position,
            heading,
            start_bearing,
        )

        print()
        print(
            "================================"
        )

        print(
            f"SEARCH TARGET "
            f"{trial_index}/"
            f"{len(TARGET_BEARINGS)}"
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
            f"Target bearing: "
            f"{start_bearing:+.1f}°"
        )

        print(
            f"Target world  : "
            f"({target[0]:+.3f}, "
            f"{target[1]:+.3f})"
        )

        print()

        #
        # The target begins outside the FOV.
        #
        # We still start with a normal observation
        # so search must be triggered from retinal
        # absence rather than an experiment flag.
        #
        phase = "SEEK"

        flyq.reset_turn_filters()

        movement_decoder = (
            MovementBurstDecoder()
        )

        center_hold = 0

        loss_count = 0

        recovery_steps = 0

        drive_neural_steps = 0
        drive_move_steps = 0

        search_direction = "LEFT"
        search_offset = 0.0

        found_count = 0

        search_events = 0
        search_steps = 0

        first_acquisition_step = None

        min_distance = float(
            "inf"
        )

        reached = False

        previous_phase = None

        trial_steps = 0

        for _ in range(
            MAX_STEPS_PER_TARGET
        ):

            global_step += 1
            trial_steps += 1

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

            (
                frame,
                target_x,
                visual_width,
                visual_height,
            ) = render_target_depth(
                relative_before,
                distance_before,
            )

            retina_mean = 0.0

            neural_direction = "CENTER"
            turn_action = "CENTER"
            turn_source = "NONE"

            filtered_turn = 0.0

            movement_hz = 0.0
            movement_state = "STOP"

            # =================================================
            # SEEK
            # =================================================

            if phase == "SEEK":

                neural = (
                    flyq.target_observation(
                        frame
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

                visible = (
                    retina_mean
                    > RETINA_PRESENT_EPS
                )

                #
                # IMPORTANT:
                # retinal darkness suppresses DRIVE.
                #
                # We do not treat black vision as
                # "CENTER, therefore walk forward."
                #
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
                            f">>> SEARCH START "
                            f"at trial step "
                            f"{trial_steps}"
                        )

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

            # =================================================
            # SEARCH
            # =================================================

            elif phase == "SEARCH":

                search_steps += 1

                neural = (
                    flyq.search_observation(
                        frame
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
                # Once retinal activity returns,
                # stop rotating immediately and
                # confirm that vision persists.
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

                        if (
                            first_acquisition_step
                            is None
                        ):

                            first_acquisition_step = (
                                trial_steps
                            )

                        phase = "SEEK"

                        loss_count = 0
                        center_hold = 0

                        flyq.reset_turn_filters()

                        print()
                        print(
                            f">>> TARGET ACQUIRED "
                            f"at trial step "
                            f"{trial_steps}"
                        )

                        print(
                            f"    retina_mean="
                            f"{retina_mean:.6f}"
                        )

                else:

                    found_count = 0

                    #
                    # ENGINEERED SCAN.
                    #
                    # This direction is determined
                    # only by internal sweep state.
                    #
                    # It NEVER checks whether the
                    # target is left or right.
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

            # =================================================
            # DRIVE
            # =================================================

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

                    loss_count = 0

                    flyq.reset_turn_filters()

            # =================================================
            # RECOVER
            # =================================================

            elif phase == "RECOVER":

                neural = (
                    flyq.target_observation(
                        frame
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
                            f">>> TARGET LOST — "
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

                "trial": trial_index,

                "trial_step": (
                    trial_steps
                ),

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

                "initial_bearing": (
                    start_bearing
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

                "visual_width": (
                    visual_width
                ),

                "visual_height": (
                    visual_height
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

                "search_offset_deg": (
                    search_offset
                ),

                "search_events": (
                    search_events
                ),
            })

            if (
                phase != previous_phase
                or
                trial_steps % 20 == 0
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
                    f"{trial_steps:>3} "
                    f"| phase="
                    f"{phase:>7} "
                    f"| x="
                    f"{x_text:>5} "
                    f"| retina="
                    f"{retina_mean:.5f} "
                    f"| turn="
                    f"{turn_action:>10} "
                    f"| move="
                    f"{movement_state:>4} "
                    f"| heading="
                    f"{heading:+6.1f}° "
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
                    f"SEARCH TARGET "
                    f"{trial_index} "
                    f"REACHED ✅"
                )

                print(
                    f"Steps             : "
                    f"{trial_steps}"
                )

                print(
                    f"First acquisition : "
                    f"{first_acquisition_step}"
                )

                print(
                    f"Search events     : "
                    f"{search_events}"
                )

                print(
                    f"Search steps      : "
                    f"{search_steps}"
                )

                print(
                    f"Distance          : "
                    f"{distance_after:.3f}"
                )

                break

        if not reached:

            print()
            print(
                f"SEARCH TARGET "
                f"{trial_index} "
                f"NOT REACHED ❌"
            )

            print(
                f"Minimum distance : "
                f"{min_distance:.3f}"
            )

        summaries.append({
            "trial": trial_index,

            "bearing": (
                start_bearing
            ),

            "success": reached,

            "steps": (
                trial_steps
            ),

            "first_acquisition": (
                first_acquisition_step
            ),

            "search_events": (
                search_events
            ),

            "search_steps": (
                search_steps
            ),

            "min_distance": (
                min_distance
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

        status = (
            "✅"
            if result[
                "success"
            ]
            else "❌"
        )

        acquisition = (
            "NONE"
            if (
                result[
                    "first_acquisition"
                ]
                is None
            )
            else str(
                result[
                    "first_acquisition"
                ]
            )
        )

        print(
            f"Trial "
            f"{result['trial']} "
            f"| bearing="
            f"{result['bearing']:+7.1f}° "
            f"| acquired="
            f"{acquisition:>4} "
            f"| search="
            f"{result['search_steps']:>3} "
            f"| steps="
            f"{result['steps']:>4} "
            f"| min_dist="
            f"{result['min_distance']:.3f} "
            f"| {status}"
        )

    print()

    print(
        f"Targets found + reached: "
        f"{successes}/"
        f"{len(TARGET_BEARINGS)}"
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
