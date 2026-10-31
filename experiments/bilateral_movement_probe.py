import numpy as np

from neural.runtime import FlyQBrainRuntime


TOTAL_ACTIVE = 800
HALF = TOTAL_ACTIVE // 2

INTENSITY = 0.5

WARMUP_STEPS = 20
STIMULUS_STEPS = 20
MEASURE_STEPS = 7

RNG = np.random.default_rng(42)


def pattern(runtime, name):
    uv = runtime.brain.uv

    x = uv[:, 0]

    n = len(x)

    light = np.zeros(
        n,
        dtype=np.float32,
    )

    order = np.argsort(x)

    if name == "black":
        return light

    if name == "left":
        chosen = order[:TOTAL_ACTIVE]

    elif name == "right":
        chosen = order[-TOTAL_ACTIVE:]

    elif name == "center":
        chosen = np.argsort(
            np.abs(x - 0.5)
        )[:TOTAL_ACTIVE]

    elif name == "bilateral_outer":
        left = order[:HALF]
        right = order[-HALF:]

        chosen = np.concatenate(
            [left, right]
        )

    elif name == "bilateral_mid":
        left_target = np.argsort(
            np.abs(x - 0.25)
        )[:HALF]

        right_target = np.argsort(
            np.abs(x - 0.75)
        )[:HALF]

        chosen = np.unique(
            np.concatenate([
                left_target,
                right_target,
            ])
        )

        # Keep exactly TOTAL_ACTIVE if overlap
        # or duplicates changed the count.
        if len(chosen) > TOTAL_ACTIVE:
            chosen = chosen[:TOTAL_ACTIVE]

    elif name == "random":
        chosen = RNG.choice(
            np.arange(n),
            size=TOTAL_ACTIVE,
            replace=False,
        )

    elif name == "full":
        chosen = np.arange(n)

    else:
        raise ValueError(name)

    light[chosen] = INTENSITY

    return light


def run(name):
    runtime = FlyQBrainRuntime()

    black = pattern(
        runtime,
        "black",
    )

    stimulus = pattern(
        runtime,
        name,
    )

    baseline_move = []

    for _ in range(WARMUP_STEPS):
        result = runtime.process_light(
            black
        )

        baseline_move.append(
            result["movement_raw_hz"]
        )

    baseline = float(
        np.mean(
            baseline_move[
                -MEASURE_STEPS:
            ]
        )
    )

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

    move_mean = float(
        np.mean(
            movement[
                -MEASURE_STEPS:
            ]
        )
    )

    turn_mean = float(
        np.mean(
            turning[
                -MEASURE_STEPS:
            ]
        )
    )

    move_delta = (
        move_mean - baseline
    )

    # Simple engineering score:
    #
    # reward movement,
    # penalize lateral turn.
    score = (
        move_delta
        - abs(turn_mean)
    )

    return {
        "name": name,

        "active": int(
            np.count_nonzero(
                stimulus
            )
        ),

        "baseline": baseline,

        "movement": move_mean,

        "delta": move_delta,

        "turn": turn_mean,

        "score": score,
    }


def main():
    conditions = [
        "black",
        "left",
        "center",
        "right",
        "bilateral_outer",
        "bilateral_mid",
        "random",
        "full",
    ]

    print()
    print(
        "FlyQ Bilateral Movement Probe"
    )
    print(
        "============================="
    )
    print()

    print(
        "Goal: high DNpe017 movement "
        "with low DNp20 turn"
    )
    print()

    results = []

    for name in conditions:

        print(
            f"Running {name}..."
        )

        result = run(name)

        results.append(result)

    print()
    print("RESULTS")
    print("=======")
    print()

    for r in results:

        print(
            f"{r['name']:>16} | "
            f"N={r['active']:>4} | "
            f"move={r['movement']:>7.2f} | "
            f"Δ={r['delta']:+7.2f} | "
            f"turn={r['turn']:+7.2f} | "
            f"score={r['score']:+7.2f}"
        )

    print()
    print("RANKING")
    print("=======")
    print(
        "Higher score = more movement "
        "and less turning"
    )
    print()

    ranked = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True,
    )

    for index, r in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"{index}. "
            f"{r['name']:<16} "
            f"score={r['score']:+7.2f}"
        )


if __name__ == "__main__":
    main()
