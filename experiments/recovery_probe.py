import numpy as np

from neural.runtime import FlyQBrainRuntime


ACTIVE_RECEPTORS = 800
INTENSITY = 0.5

WARMUP_STEPS = 20
STIMULUS_STEPS = 14

MAX_RECOVERY_STEPS = 100

# Require CENTER for this many consecutive steps
# before declaring the response recovered.
STABLE_CENTER_STEPS = 7


def make_light(runtime, condition):
    x = runtime.brain.uv[:, 0]

    light = np.zeros(
        len(x),
        dtype=np.float32,
    )

    if condition == "black":
        return light

    if condition == "left":
        indices = np.argsort(x)[
            :ACTIVE_RECEPTORS
        ]

    elif condition == "right":
        indices = np.argsort(x)[
            -ACTIVE_RECEPTORS:
        ]

    else:
        raise ValueError(condition)

    light[indices] = INTENSITY

    return light


def run_trial(condition):
    runtime = FlyQBrainRuntime()

    black = make_light(
        runtime,
        "black",
    )

    stimulus = make_light(
        runtime,
        condition,
    )

    print()
    print(
        f"========== {condition.upper()} =========="
    )

    #
    # 1. Dark adaptation / baseline
    #

    baseline = []

    for _ in range(WARMUP_STEPS):
        result = runtime.process_light(
            black
        )

        baseline.append(
            result["turn_raw_hz"]
        )

    baseline_mean = float(
        np.mean(baseline[-7:])
    )

    print(
        f"Baseline raw mean "
        f"(last 7): {baseline_mean:.2f} Hz"
    )

    #
    # 2. Stimulus
    #

    print()
    print("STIMULUS")
    print("--------")

    last_stimulus = None

    for step in range(
        1,
        STIMULUS_STEPS + 1,
    ):
        result = runtime.process_light(
            stimulus
        )

        last_stimulus = result

        print(
            f"{step:>3} | "
            f"raw={result['turn_raw_hz']:>7.2f} | "
            f"avg={result['filtered_turn_hz']:>7.2f} | "
            f"{result['direction']}"
        )

    print()
    print(
        "Stimulus removed -> BLACK"
    )
    print()

    #
    # 3. Recovery
    #

    first_center = None
    stable_center = None

    consecutive_center = 0

    for step in range(
        1,
        MAX_RECOVERY_STEPS + 1,
    ):
        result = runtime.process_light(
            black
        )

        if result["direction"] == "CENTER":
            consecutive_center += 1

            if first_center is None:
                first_center = step

        else:
            consecutive_center = 0

        print(
            f"{step:>3} | "
            f"{step * runtime.step_ms:>7.1f} ms | "
            f"raw={result['turn_raw_hz']:>7.2f} | "
            f"avg={result['filtered_turn_hz']:>7.2f} | "
            f"{result['direction']}"
        )

        if (
            consecutive_center
            >= STABLE_CENTER_STEPS
        ):
            stable_center = (
                step
                - STABLE_CENTER_STEPS
                + 1
            )

            break

    print()
    print("RECOVERY SUMMARY")
    print("----------------")

    if first_center is None:
        print(
            "First CENTER : not reached"
        )
    else:
        print(
            f"First CENTER : step "
            f"{first_center} "
            f"(~{first_center * runtime.step_ms:.1f} ms)"
        )

    if stable_center is None:
        print(
            "Stable CENTER: not reached"
        )
    else:
        print(
            f"Stable CENTER: step "
            f"{stable_center} "
            f"(~{stable_center * runtime.step_ms:.1f} ms)"
        )

    print(
        f"Last stimulus raw: "
        f"{last_stimulus['turn_raw_hz']:.2f} Hz"
    )


def main():
    print()
    print("FlyQ Neural Recovery Probe")
    print("==========================")
    print()
    print(
        f"Stimulus receptors : {ACTIVE_RECEPTORS}"
    )
    print(
        f"Stimulus intensity : {INTENSITY}"
    )
    print(
        f"Neural step        : 28.6 ms"
    )

    run_trial("left")
    run_trial("right")


if __name__ == "__main__":
    main()
