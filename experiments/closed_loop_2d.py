import csv
import math
from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime
from control.movement_decoder import MovementDecoder


# --------------------------------------------------
# Neural stimulation
# --------------------------------------------------

ACTIVE_DIRECTION = 800
ACTIVE_FORWARD_PER_SIDE = 400
INTENSITY = 0.5

# --------------------------------------------------
# Simulated body
# --------------------------------------------------

MOVE_STEP = 0.01

TURN_STEP_DEG = 2.0
TURN_STEP = math.radians(TURN_STEP_DEG)

ANGLE_TOLERANCE_DEG = 12.0
ANGLE_TOLERANCE = math.radians(
    ANGLE_TOLERANCE_DEG
)

TARGET_RADIUS = 0.10

STABLE_STEPS_REQUIRED = 5
MAX_STEPS_PER_TARGET = 600

# Continuous trajectory.
# We do NOT reset the brain or body between targets.
TARGETS = [
    (0.60, 0.40),
    (-0.30, 0.70),
    (-0.70, -0.20),
    (0.30, -0.60),
    (0.00, 0.00),
]

OUTPUT = Path(
    "telemetry/closed_loop_2d.csv"
)


def wrap_angle(angle):
    """
    Normalize angle to [-pi, +pi].
    """
    return (
        angle + math.pi
    ) % (
        2 * math.pi
    ) - math.pi


def create_patterns(runtime):
    x = runtime.brain.uv[:, 0]

    order = np.argsort(x)

    black = np.zeros(
        len(x),
        dtype=np.float32,
    )

    left = black.copy()

    left[
        order[:ACTIVE_DIRECTION]
    ] = INTENSITY

    right = black.copy()

    right[
        order[-ACTIVE_DIRECTION:]
    ] = INTENSITY

    forward = black.copy()

    forward[
        order[:ACTIVE_FORWARD_PER_SIDE]
    ] = INTENSITY

    forward[
        order[-ACTIVE_FORWARD_PER_SIDE:]
    ] = INTENSITY

    return {
        "black": black,
        "left": left,
        "right": right,
        "forward": forward,
    }


def geometry(
    x,
    y,
    heading,
    target_x,
    target_y,
):
    dx = target_x - x
    dy = target_y - y

    distance = math.hypot(
        dx,
        dy,
    )

    bearing = math.atan2(
        dy,
        dx,
    )

    angle_error = wrap_angle(
        bearing - heading
    )

    return (
        distance,
        angle_error,
    )


def choose_sensory(
    distance,
    angle_error,
):
    #
    # When we reach the target zone, remove
    # the visual drive and let the network
    # naturally decay toward STOP.
    #
    if distance <= TARGET_RADIUS:
        return "black"

    #
    # Positive angle error = target lies
    # counter-clockwise / LEFT.
    #
    if angle_error > ANGLE_TOLERANCE:
        return "left"

    #
    # Negative angle error = target lies
    # clockwise / RIGHT.
    #
    if angle_error < -ANGLE_TOLERANCE:
        return "right"

    #
    # Target approximately ahead:
    # bilateral outer stimulation.
    #
    return "forward"


def main():
    runtime = FlyQBrainRuntime()

    movement_decoder = MovementDecoder(
        on_threshold=30.0,
        off_threshold=25.0,
        on_samples=3,
        off_samples=2,
    )

    patterns = create_patterns(
        runtime
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = 0.0
    y = 0.0

    # Facing east initially.
    heading = 0.0

    print()
    print("FlyQ 2D Closed-Loop Navigation")
    print("==============================")
    print()
    print(
        f"MOVE step : {MOVE_STEP}"
    )
    print(
        f"TURN step : {TURN_STEP_DEG:.1f}°"
    )
    print(
        f"Target radius : {TARGET_RADIUS}"
    )
    print()

    #
    # Neural warmup.
    #
    print("Neural warmup...")

    for _ in range(30):
        result = runtime.process_light(
            patterns["black"]
        )

        movement_decoder.update(
            result["movement_raw_hz"]
        )

    print("Ready.")
    print()

    successes = 0

    with OUTPUT.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "target_number",
            "step",
            "x",
            "y",
            "heading_deg",
            "target_x",
            "target_y",
            "distance",
            "angle_error_deg",
            "sensory",
            "turn_raw_hz",
            "turn_filtered_hz",
            "direction",
            "movement_hz",
            "movement_state",
        ])

        for target_number, (
            target_x,
            target_y,
        ) in enumerate(
            TARGETS,
            start=1,
        ):

            print(
                f"===== TARGET {target_number} "
                f"({target_x:+.2f}, "
                f"{target_y:+.2f}) ====="
            )

            stable = 0

            last_sensory = None
            last_direction = None
            last_move = None

            for step in range(
                1,
                MAX_STEPS_PER_TARGET + 1,
            ):

                distance_before, angle_error = (
                    geometry(
                        x,
                        y,
                        heading,
                        target_x,
                        target_y,
                    )
                )

                sensory = choose_sensory(
                    distance_before,
                    angle_error,
                )

                result = runtime.process_light(
                    patterns[sensory]
                )

                move_decision = (
                    movement_decoder.update(
                        result[
                            "movement_raw_hz"
                        ]
                    )
                )

                direction = result[
                    "direction"
                ]

                #
                # Apply neural turning.
                #
                if result["ready"]:

                    if direction == "LEFT":
                        heading += TURN_STEP

                    elif direction == "RIGHT":
                        heading -= TURN_STEP

                heading = wrap_angle(
                    heading
                )

                #
                # Apply neural movement.
                #
                if (
                    result["ready"]
                    and move_decision.state
                    == "MOVE"
                ):
                    x += (
                        math.cos(heading)
                        * MOVE_STEP
                    )

                    y += (
                        math.sin(heading)
                        * MOVE_STEP
                    )

                distance_after, new_angle_error = (
                    geometry(
                        x,
                        y,
                        heading,
                        target_x,
                        target_y,
                    )
                )

                #
                # A target counts only when:
                #
                # - body is physically close
                # - movement has stopped
                # - turning decoder is CENTER
                #
                if (
                    distance_after
                    <= TARGET_RADIUS
                    and move_decision.state
                    == "STOP"
                    and direction
                    == "CENTER"
                ):
                    stable += 1
                else:
                    stable = 0

                writer.writerow([
                    target_number,
                    step,
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{math.degrees(heading):.6f}",
                    f"{target_x:.6f}",
                    f"{target_y:.6f}",
                    f"{distance_after:.6f}",
                    f"{math.degrees(new_angle_error):.6f}",
                    sensory,
                    f"{result['turn_raw_hz']:.6f}",
                    f"{result['filtered_turn_hz']:.6f}",
                    direction,
                    f"{result['movement_raw_hz']:.6f}",
                    move_decision.state,
                ])

                #
                # Keep console output manageable.
                #
                state_changed = (
                    sensory != last_sensory
                    or direction
                    != last_direction
                    or move_decision.state
                    != last_move
                )

                if (
                    step == 1
                    or step % 10 == 0
                    or state_changed
                    or distance_after
                    <= TARGET_RADIUS
                ):
                    print(
                        f"{step:>3} | "
                        f"sense={sensory:>7} | "
                        f"dir={direction:>6} | "
                        f"move={move_decision.state:>4} | "
                        f"pos=({x:+.2f},{y:+.2f}) | "
                        f"head={math.degrees(heading):+6.1f}° | "
                        f"dist={distance_after:.3f} | "
                        f"err={math.degrees(new_angle_error):+6.1f}°"
                    )

                last_sensory = sensory
                last_direction = direction
                last_move = (
                    move_decision.state
                )

                if (
                    stable
                    >= STABLE_STEPS_REQUIRED
                ):
                    successes += 1

                    print()
                    print(
                        f"TARGET ACQUIRED ✅ "
                        f"after {step} neural steps"
                    )
                    print()

                    break

            else:
                print()
                print(
                    "TARGET NOT STABILIZED ❌"
                )
                print()

    print(
        "=============================="
    )

    print(
        f"Targets acquired: "
        f"{successes}/{len(TARGETS)}"
    )

    print(
        f"Final position : "
        f"({x:+.3f}, {y:+.3f})"
    )

    print(
        f"Final heading  : "
        f"{math.degrees(heading):+.1f}°"
    )

    print()
    print(
        f"Telemetry saved → {OUTPUT}"
    )


if __name__ == "__main__":
    main()
