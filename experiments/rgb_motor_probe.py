import numpy as np

from neural.runtime import FlyQBrainRuntime


WIDTH = 320
HEIGHT = 240

# sRGB ~188 corresponds approximately
# to 0.5 linear luminance.
PIXEL = 188

WARMUP_STEPS = 20
STIMULUS_STEPS = 20
MEASURE_STEPS = 7


def black_frame():
    return np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )


def make_frame(name):
    frame = black_frame()

    if name == "black":
        return frame

    if name == "left_half":
        frame[:, : WIDTH // 2] = PIXEL

    elif name == "right_half":
        frame[:, WIDTH // 2 :] = PIXEL

    elif name == "center_third":
        x1 = WIDTH // 3
        x2 = 2 * WIDTH // 3

        frame[:, x1:x2] = PIXEL

    elif name.startswith("outer_"):
        fraction = float(
            name.split("_")[1]
        ) / 100.0

        band = int(
            WIDTH * fraction
        )

        frame[:, :band] = PIXEL
        frame[:, WIDTH - band :] = PIXEL

    elif name == "full":
        frame[:] = PIXEL

    else:
        raise ValueError(name)

    return frame


def run(name):
    runtime = FlyQBrainRuntime()

    black = black_frame()
    stimulus = make_frame(name)

    baseline_move = []
    baseline_turn = []

    for _ in range(WARMUP_STEPS):
        result = runtime.process_frame(
            black
        )

        baseline_move.append(
            result["movement_raw_hz"]
        )

        baseline_turn.append(
            result["turn_raw_hz"]
        )

    baseline_move_mean = float(
        np.mean(
            baseline_move[
                -MEASURE_STEPS:
            ]
        )
    )

    baseline_turn_mean = float(
        np.mean(
            baseline_turn[
                -MEASURE_STEPS:
            ]
        )
    )

    movement = []
    turning = []
    retina = []

    for _ in range(STIMULUS_STEPS):
        result = runtime.process_frame(
            stimulus
        )

        movement.append(
            result["movement_raw_hz"]
        )

        turning.append(
            result["turn_raw_hz"]
        )

        retina.append(
            result["retina_mean"]
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

    retina_mean = float(
        np.mean(
            retina[
                -MEASURE_STEPS:
            ]
        )
    )

    move_delta = (
        move_mean
        - baseline_move_mean
    )

    turn_shift = (
        turn_mean
        - baseline_turn_mean
    )

    #
    # Forward-pattern score:
    #
    # reward movement increase,
    # penalize CHANGE in turn compared
    # with the natural baseline bias.
    #
    score = (
        move_delta
        - abs(turn_shift)
    )

    return {
        "name": name,

        "retina": retina_mean,

        "baseline_move": (
            baseline_move_mean
        ),

        "baseline_turn": (
            baseline_turn_mean
        ),

        "move": move_mean,

        "move_delta": move_delta,

        "turn": turn_mean,

        "turn_shift": turn_shift,

        "score": score,
    }


def main():
    conditions = [
        "black",

        "left_half",
        "right_half",
        "center_third",

        "outer_10",
        "outer_15",
        "outer_20",
        "outer_25",
        "outer_30",

        "full",
    ]

    print()
    print("FlyQ RGB Motor Bridge Probe")
    print("===========================")
    print()
    print(
        "Testing actual RGB → retina → MaleCNS"
    )
    print()

    results = []

    for name in conditions:
        print(
            f"Running {name}..."
        )

        results.append(
            run(name)
        )

    print()
    print("RESULTS")
    print("=======")
    print()

    for r in results:
        print(
            f"{r['name']:>14} | "
            f"retina={r['retina']:.3f} | "
            f"move={r['move']:>7.2f} | "
            f"Δmove={r['move_delta']:+7.2f} | "
            f"turn={r['turn']:+7.2f} | "
            f"Δturn={r['turn_shift']:+7.2f} | "
            f"score={r['score']:+7.2f}"
        )

    print()
    print("FORWARD PATTERN RANKING")
    print("=======================")
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
            f"{index:>2}. "
            f"{r['name']:<14} "
            f"score={r['score']:+7.2f}"
        )


if __name__ == "__main__":
    main()
