import numpy as np

from neural.runtime import FlyQBrainRuntime


ACTIVE_RECEPTORS = 800
INTENSITY = 0.5

MOVE_STEP = 0.01
TARGET_TOLERANCE = 0.10

MAX_STEPS_PER_TARGET = 400
STABLE_STEPS_REQUIRED = 5

TARGETS = [
    -0.80,
    +0.80,
    -0.40,
    +0.40,
    0.00,
]


def retinal_regions(runtime):
    x = runtime.brain.uv[:, 0]

    return {
        "left": np.argsort(x)[:ACTIVE_RECEPTORS],

        "center": np.argsort(
            np.abs(x - 0.5)
        )[:ACTIVE_RECEPTORS],

        "right": np.argsort(x)[-ACTIVE_RECEPTORS:],
    }


def make_light(runtime, regions, condition):
    light = np.zeros(
        len(runtime.brain.uv),
        dtype=np.float32,
    )

    if condition != "black":
        light[regions[condition]] = INTENSITY

    return light


def sensory_condition(position, target):
    error = target - position

    if error < -TARGET_TOLERANCE:
        return "left"

    if error > TARGET_TOLERANCE:
        return "right"

    return "center"


def apply_action(position, direction):
    if direction == "LEFT":
        position -= MOVE_STEP

    elif direction == "RIGHT":
        position += MOVE_STEP

    return float(
        np.clip(position, -1.0, 1.0)
    )


def main():
    runtime = FlyQBrainRuntime()

    regions = retinal_regions(runtime)

    position = 0.0

    print()
    print("FlyQ 1D Closed-Loop Navigation")
    print("==============================")
    print()

    #
    # Let the fresh brain settle before beginning.
    #
    black = make_light(
        runtime,
        regions,
        "black",
    )

    print("Neural warmup...")

    for _ in range(20):
        runtime.process_light(black)

    print("Ready.")
    print()

    successes = 0

    for target_number, target in enumerate(
        TARGETS,
        start=1,
    ):
        print()
        print(
            f"===== TARGET {target_number}: "
            f"{target:+.2f} ====="
        )

        stable_steps = 0

        for step in range(
            1,
            MAX_STEPS_PER_TARGET + 1,
        ):
            error_before = target - position

            sensory = sensory_condition(
                position,
                target,
            )

            light = make_light(
                runtime,
                regions,
                sensory,
            )

            result = runtime.process_light(
                light
            )

            old_position = position

            if result["ready"]:
                position = apply_action(
                    position,
                    result["direction"],
                )

            error_after = target - position

            aligned = (
                abs(error_after)
                <= TARGET_TOLERANCE
            )

            if (
                aligned
                and result["direction"]
                == "CENTER"
            ):
                stable_steps += 1
            else:
                stable_steps = 0

            print(
                f"{step:>3} | "
                f"sense={sensory:>6} | "
                f"raw={result['turn_raw_hz']:>7.2f} | "
                f"avg={result['filtered_turn_hz']:>7.2f} | "
                f"action={result['direction']:>6} | "
                f"pos={old_position:+.2f}"
                f"->{position:+.2f} | "
                f"target={target:+.2f} | "
                f"err={error_after:+.2f}"
            )

            if (
                stable_steps
                >= STABLE_STEPS_REQUIRED
            ):
                successes += 1

                print()
                print(
                    f"TARGET ACQUIRED ✅ "
                    f"after {step} neural steps"
                )

                break

        else:
            print()
            print(
                "TARGET NOT STABILIZED ❌"
            )

    print()
    print("==============================")
    print(
        f"Targets acquired: "
        f"{successes}/{len(TARGETS)}"
    )
    print(
        f"Final position : {position:+.2f}"
    )


if __name__ == "__main__":
    main()
