import numpy as np

from neural.runtime import FlyQBrainRuntime
from control.movement_decoder import MovementDecoder


ACTIVE_PER_SIDE = 400
INTENSITY = 0.5

BASELINE_STEPS = 30
STIMULUS_STEPS = 30
RECOVERY_STEPS = 30


def black_light(runtime):
    return np.zeros(
        len(runtime.brain.uv),
        dtype=np.float32,
    )


def bilateral_outer(runtime):
    x = runtime.brain.uv[:, 0]
    order = np.argsort(x)

    light = np.zeros(
        len(x),
        dtype=np.float32,
    )

    light[
        order[:ACTIVE_PER_SIDE]
    ] = INTENSITY

    light[
        order[-ACTIVE_PER_SIDE:]
    ] = INTENSITY

    return light


def show(
    phase,
    step,
    runtime,
    result,
    decision,
):
    print(
        f"{phase:<8} | "
        f"{step:>3} | "
        f"{step * runtime.step_ms:>7.1f} ms | "
        f"DNpe017={result['movement_raw_hz']:>7.2f} | "
        f"turn={result['turn_raw_hz']:>7.2f} | "
        f"{decision.state:<4} | "
        f"on={decision.on_count} "
        f"off={decision.off_count}"
    )


def main():
    runtime = FlyQBrainRuntime()

    decoder = MovementDecoder(
        on_threshold=30.0,
        off_threshold=25.0,
        on_samples=3,
        off_samples=2,
    )

    black = black_light(runtime)
    stimulus = bilateral_outer(runtime)

    print()
    print("FlyQ Movement Decoder Verification")
    print("=================================")
    print()
    print("MOVE : 3 consecutive >= 30 Hz")
    print("STOP : 2 consecutive < 25 Hz")
    print()

    #
    # Baseline
    #
    for step in range(
        1,
        BASELINE_STEPS + 1,
    ):
        result = runtime.process_light(
            black
        )

        decision = decoder.update(
            result["movement_raw_hz"]
        )

        show(
            "BASE",
            step,
            runtime,
            result,
            decision,
        )

    #
    # Bilateral stimulus
    #
    print()
    print(">>> BILATERAL OUTER ON")
    print()

    for step in range(
        1,
        STIMULUS_STEPS + 1,
    ):
        result = runtime.process_light(
            stimulus
        )

        decision = decoder.update(
            result["movement_raw_hz"]
        )

        show(
            "ON",
            step,
            runtime,
            result,
            decision,
        )

    #
    # Remove stimulus
    #
    print()
    print(">>> STIMULUS OFF")
    print()

    for step in range(
        1,
        RECOVERY_STEPS + 1,
    ):
        result = runtime.process_light(
            black
        )

        decision = decoder.update(
            result["movement_raw_hz"]
        )

        show(
            "OFF",
            step,
            runtime,
            result,
            decision,
        )


if __name__ == "__main__":
    main()
