import numpy as np

from neural.runtime import FlyQBrainRuntime


ACTIVE_PER_SIDE = 400
INTENSITY = 0.5

WARMUP_STEPS = 30
STIMULUS_STEPS = 30
RECOVERY_STEPS = 50

# Provisional only.
# We are testing whether this value cleanly separates
# baseline from bilateral-driven activity.
CANDIDATE_MOVE_THRESHOLD = 30.0

STABLE_STEPS = 5


def black_light(runtime):
    return np.zeros(
        len(runtime.brain.uv),
        dtype=np.float32,
    )


def bilateral_outer(runtime):
    x = runtime.brain.uv[:, 0]
    order = np.argsort(x)

    left = order[:ACTIVE_PER_SIDE]
    right = order[-ACTIVE_PER_SIDE:]

    light = np.zeros(
        len(x),
        dtype=np.float32,
    )

    light[left] = INTENSITY
    light[right] = INTENSITY

    return light


def main():
    runtime = FlyQBrainRuntime()

    black = black_light(runtime)
    stimulus = bilateral_outer(runtime)

    print()
    print("FlyQ DNpe017 Temporal Probe")
    print("===========================")
    print()
    print(
        "Stimulus: bilateral_outer "
        f"({ACTIVE_PER_SIDE} + "
        f"{ACTIVE_PER_SIDE} receptors)"
    )
    print(
        f"Candidate MOVE threshold: "
        f"{CANDIDATE_MOVE_THRESHOLD:.1f} Hz"
    )
    print()

    #
    # Warmup / baseline
    #

    baseline = []

    print("BASELINE")
    print("--------")

    for step in range(
        1,
        WARMUP_STEPS + 1,
    ):
        result = runtime.process_light(
            black
        )

        movement = result[
            "movement_raw_hz"
        ]

        baseline.append(movement)

        print(
            f"{step:>3} | "
            f"move={movement:>7.2f} | "
            f"turn={result['turn_raw_hz']:>7.2f}"
        )

    baseline_last = np.asarray(
        baseline[-10:],
        dtype=float,
    )

    baseline_mean = float(
        baseline_last.mean()
    )

    baseline_std = float(
        baseline_last.std()
    )

    print()
    print(
        f"Baseline last-10 mean: "
        f"{baseline_mean:.2f} Hz"
    )

    print(
        f"Baseline last-10 std : "
        f"{baseline_std:.2f} Hz"
    )

    #
    # Stimulus onset
    #

    print()
    print("STIMULUS ON")
    print("-----------")

    first_move = None
    consecutive_move = 0
    stable_move = None

    stimulus_values = []

    for step in range(
        1,
        STIMULUS_STEPS + 1,
    ):
        result = runtime.process_light(
            stimulus
        )

        movement = result[
            "movement_raw_hz"
        ]

        stimulus_values.append(
            movement
        )

        moving = (
            movement
            >= CANDIDATE_MOVE_THRESHOLD
        )

        if moving:
            if first_move is None:
                first_move = step

            consecutive_move += 1
        else:
            consecutive_move = 0

        if (
            stable_move is None
            and consecutive_move >= STABLE_STEPS
        ):
            stable_move = (
                step
                - STABLE_STEPS
                + 1
            )

        print(
            f"{step:>3} | "
            f"{step * runtime.step_ms:>7.1f} ms | "
            f"move={movement:>7.2f} | "
            f"turn={result['turn_raw_hz']:>7.2f} | "
            f"{'MOVE' if moving else 'STOP'}"
        )

    #
    # Stimulus removed
    #

    print()
    print("STIMULUS OFF -> BLACK")
    print("---------------------")

    first_stop = None
    consecutive_stop = 0
    stable_stop = None

    for step in range(
        1,
        RECOVERY_STEPS + 1,
    ):
        result = runtime.process_light(
            black
        )

        movement = result[
            "movement_raw_hz"
        ]

        moving = (
            movement
            >= CANDIDATE_MOVE_THRESHOLD
        )

        if not moving:
            if first_stop is None:
                first_stop = step

            consecutive_stop += 1
        else:
            consecutive_stop = 0

        print(
            f"{step:>3} | "
            f"{step * runtime.step_ms:>7.1f} ms | "
            f"move={movement:>7.2f} | "
            f"turn={result['turn_raw_hz']:>7.2f} | "
            f"{'MOVE' if moving else 'STOP'}"
        )

        if (
            consecutive_stop
            >= STABLE_STEPS
        ):
            stable_stop = (
                step
                - STABLE_STEPS
                + 1
            )

            break

    stimulus_tail = float(
        np.mean(
            stimulus_values[-10:]
        )
    )

    print()
    print("SUMMARY")
    print("=======")

    print(
        f"Baseline mean       : "
        f"{baseline_mean:.2f} Hz"
    )

    print(
        f"Late stimulus mean  : "
        f"{stimulus_tail:.2f} Hz"
    )

    print(
        f"Stimulus Δ          : "
        f"{stimulus_tail - baseline_mean:+.2f} Hz"
    )

    if first_move is not None:
        print(
            f"First MOVE          : "
            f"step {first_move} "
            f"(~{first_move * runtime.step_ms:.1f} ms)"
        )

    if stable_move is not None:
        print(
            f"Stable MOVE         : "
            f"step {stable_move} "
            f"(~{stable_move * runtime.step_ms:.1f} ms)"
        )

    if first_stop is not None:
        print(
            f"First STOP          : "
            f"step {first_stop} "
            f"(~{first_stop * runtime.step_ms:.1f} ms)"
        )

    if stable_stop is not None:
        print(
            f"Stable STOP         : "
            f"step {stable_stop} "
            f"(~{stable_stop * runtime.step_ms:.1f} ms)"
        )


if __name__ == "__main__":
    main()
