import numpy as np

from neural.runtime import FlyQBrainRuntime


WIDTH = 320
HEIGHT = 240

PIXEL = 188

STRIPE_WIDTH = 16
PERIOD = 32

SHIFT_PX = 4

WARMUP_STEPS = 30
STIMULUS_STEPS = 60

IGNORE_FIRST = 10


def black_frame():
    return np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )


def grating_frame(phase):
    frame = black_frame()

    xs = np.arange(WIDTH)

    mask = (
        ((xs + phase) % PERIOD)
        < STRIPE_WIDTH
    )

    frame[:, mask, :] = PIXEL

    return frame


def run(condition):
    runtime = FlyQBrainRuntime()

    black = black_frame()

    #
    # Dark baseline
    #
    baseline_turn = []
    baseline_move = []

    for _ in range(WARMUP_STEPS):

        result = runtime.process_frame(
            black
        )

        baseline_turn.append(
            result["turn_raw_hz"]
        )

        baseline_move.append(
            result["movement_raw_hz"]
        )

    baseline_turn = float(
        np.mean(
            baseline_turn[-10:]
        )
    )

    baseline_move = float(
        np.mean(
            baseline_move[-10:]
        )
    )

    turns = []
    movements = []
    retina = []

    phase = 0

    for step in range(
        STIMULUS_STEPS
    ):

        if condition == "static":
            phase = 0

        elif condition == "flow_left":
            phase -= SHIFT_PX

        elif condition == "flow_right":
            phase += SHIFT_PX

        else:
            raise ValueError(
                condition
            )

        frame = grating_frame(
            phase
        )

        result = runtime.process_frame(
            frame
        )

        turns.append(
            result["turn_raw_hz"]
        )

        movements.append(
            result["movement_raw_hz"]
        )

        retina.append(
            result["retina_mean"]
        )

    turns_eval = np.asarray(
        turns[IGNORE_FIRST:],
        dtype=float,
    )

    moves_eval = np.asarray(
        movements[IGNORE_FIRST:],
        dtype=float,
    )

    retina_eval = np.asarray(
        retina[IGNORE_FIRST:],
        dtype=float,
    )

    return {
        "condition": condition,

        "baseline_turn": baseline_turn,
        "baseline_move": baseline_move,

        "turn_mean": float(
            turns_eval.mean()
        ),

        "turn_std": float(
            turns_eval.std()
        ),

        "move_mean": float(
            moves_eval.mean()
        ),

        "move_std": float(
            moves_eval.std()
        ),

        "retina_mean": float(
            retina_eval.mean()
        ),

        "turn_series": turns,
    }


def main():
    print()
    print("FlyQ RGB Motion Direction Probe")
    print("==============================")
    print()
    print(
        "Same luminance."
    )
    print(
        "Only visual motion direction changes."
    )
    print()

    conditions = [
        "static",
        "flow_left",
        "flow_right",
    ]

    results = {}

    for condition in conditions:

        print(
            f"Running {condition}..."
        )

        results[condition] = run(
            condition
        )

    print()
    print("RESULTS")
    print("=======")
    print()

    for condition in conditions:

        r = results[condition]

        print(
            f"{condition:>10} | "
            f"retina={r['retina_mean']:.3f} | "
            f"turn={r['turn_mean']:+7.2f} "
            f"± {r['turn_std']:.2f} | "
            f"move={r['move_mean']:7.2f} "
            f"± {r['move_std']:.2f}"
        )

    static_turn = (
        results["static"][
            "turn_mean"
        ]
    )

    left_turn = (
        results["flow_left"][
            "turn_mean"
        ]
    )

    right_turn = (
        results["flow_right"][
            "turn_mean"
        ]
    )

    print()
    print("MOTION EFFECT")
    print("=============")
    print()

    print(
        f"LEFT flow  vs static: "
        f"{left_turn - static_turn:+.2f} Hz"
    )

    print(
        f"RIGHT flow vs static: "
        f"{right_turn - static_turn:+.2f} Hz"
    )

    print(
        f"RIGHT - LEFT separation: "
        f"{right_turn - left_turn:+.2f} Hz"
    )

    print()
    print("First 20 turn samples")
    print("---------------------")

    for condition in conditions:

        values = (
            results[condition][
                "turn_series"
            ][:20]
        )

        formatted = " ".join(
            f"{v:+.1f}"
            for v in values
        )

        print(
            f"{condition:>10}: "
            f"{formatted}"
        )


if __name__ == "__main__":
    main()
