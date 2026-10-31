import numpy as np

from neural.runtime import FlyQBrainRuntime


ACTIVE_RECEPTORS = 800

WARMUP_STEPS = 20
STIMULUS_STEPS = 20
MEASURE_STEPS = 7

INTENSITIES = [
    0.25,
    0.50,
    1.00,
]


def make_light(runtime, condition, intensity):
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

    elif condition == "center":
        indices = np.argsort(
            np.abs(x - 0.5)
        )[:ACTIVE_RECEPTORS]

    elif condition == "full":
        indices = np.arange(len(x))

    else:
        raise ValueError(condition)

    light[indices] = intensity

    return light


def run_condition(condition, intensity):
    #
    # Fresh identical brain for every condition.
    #
    runtime = FlyQBrainRuntime()

    black = make_light(
        runtime,
        "black",
        0.0,
    )

    stimulus = make_light(
        runtime,
        condition,
        intensity,
    )

    #
    # Establish dark baseline.
    #
    baseline = []

    for _ in range(WARMUP_STEPS):
        result = runtime.process_light(
            black
        )

        baseline.append(
            result["movement_raw_hz"]
        )

    baseline_move = float(
        np.mean(
            baseline[-MEASURE_STEPS:]
        )
    )

    #
    # Apply controlled visual stimulus.
    #
    movement = []
    turning = []

    for _ in range(STIMULUS_STEPS):
        result = runtime.process_light(
            stimulus
        )

        movement.append(
            result["movement_raw_hz"]
        )

        turning.append(
            result["turn_raw_hz"]
        )

    movement_mean = float(
        np.mean(
            movement[-MEASURE_STEPS:]
        )
    )

    turn_mean = float(
        np.mean(
            turning[-MEASURE_STEPS:]
        )
    )

    return {
        "condition": condition,
        "intensity": intensity,

        "baseline_move": baseline_move,

        "movement_mean": movement_mean,

        "movement_delta": (
            movement_mean
            - baseline_move
        ),

        "turn_mean": turn_mean,

        "active_receptors": int(
            np.count_nonzero(stimulus)
        ),
    }


def main():
    print()
    print("FlyQ DNpe017 Movement Probe")
    print("===========================")
    print()
    print(
        "Movement signal = "
        "DNpe017_L + DNpe017_R"
    )
    print()

    results = []

    #
    # Pure dark control.
    #
    black = run_condition(
        "black",
        0.0,
    )

    results.append(black)

    print(
        f"BLACK | "
        f"baseline={black['baseline_move']:7.2f} | "
        f"move={black['movement_mean']:7.2f} | "
        f"Δ={black['movement_delta']:+7.2f}"
    )

    print()

    for intensity in INTENSITIES:

        print(
            f"=== intensity {intensity:.2f} ==="
        )

        for condition in [
            "left",
            "center",
            "right",
            "full",
        ]:

            result = run_condition(
                condition,
                intensity,
            )

            results.append(result)

            print(
                f"{condition:>6} | "
                f"N={result['active_receptors']:>4} | "
                f"baseline={result['baseline_move']:7.2f} | "
                f"move={result['movement_mean']:7.2f} | "
                f"Δ={result['movement_delta']:+7.2f} | "
                f"turn={result['turn_mean']:+7.2f}"
            )

        print()

    print()
    print("SUMMARY")
    print("=======")
    print()
    print(
        "Positive Δ means visual stimulation "
        "increased DNpe017 summed activity."
    )


if __name__ == "__main__":
    main()
