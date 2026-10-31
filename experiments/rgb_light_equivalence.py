import sys
from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime


UPSTREAM = Path.home() / "FlyQ" / "upstream-doomfly"

if str(UPSTREAM) not in sys.path:
    sys.path.insert(
        0,
        str(UPSTREAM),
    )

from doom.game import retinal_samples


WIDTH = 320
HEIGHT = 240

PIXEL = 188

TARGET_WIDTH = 64
TARGET_HEIGHT = 160

WARMUP_STEPS = 20
STIMULUS_STEPS = 20

POSITIONS = [
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
]


def black_frame():
    return np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )


def target_frame(x_fraction):
    frame = black_frame()

    cx = int(
        x_fraction * (WIDTH - 1)
    )

    cy = HEIGHT // 2

    x1 = max(
        0,
        cx - TARGET_WIDTH // 2,
    )

    x2 = min(
        WIDTH,
        cx + TARGET_WIDTH // 2,
    )

    y1 = max(
        0,
        cy - TARGET_HEIGHT // 2,
    )

    y2 = min(
        HEIGHT,
        cy + TARGET_HEIGHT // 2,
    )

    frame[
        y1:y2,
        x1:x2,
    ] = PIXEL

    return frame


def run_position(x_fraction):
    #
    # Two completely fresh brains.
    #
    rgb_runtime = FlyQBrainRuntime()
    light_runtime = FlyQBrainRuntime()

    black = black_frame()

    stimulus = target_frame(
        x_fraction
    )

    #
    # Explicitly produce exactly the receptor
    # vectors process_frame() should receive.
    #
    black_light = retinal_samples(
        black,
        light_runtime.brain.uv,
    )

    stimulus_light = retinal_samples(
        stimulus,
        light_runtime.brain.uv,
    )

    max_turn_error = 0.0
    max_move_error = 0.0

    #
    # Identical baseline histories.
    #
    for step in range(
        1,
        WARMUP_STEPS + 1,
    ):
        rgb = rgb_runtime.process_frame(
            black
        )

        direct = light_runtime.process_light(
            black_light
        )

        turn_error = abs(
            rgb["turn_raw_hz"]
            - direct["turn_raw_hz"]
        )

        move_error = abs(
            rgb["movement_raw_hz"]
            - direct["movement_raw_hz"]
        )

        max_turn_error = max(
            max_turn_error,
            turn_error,
        )

        max_move_error = max(
            max_move_error,
            move_error,
        )

    rgb_turn = []
    direct_turn = []

    rgb_move = []
    direct_move = []

    print()
    print(
        f"--- x={x_fraction:.2f} ---"
    )

    for step in range(
        1,
        STIMULUS_STEPS + 1,
    ):
        rgb = rgb_runtime.process_frame(
            stimulus
        )

        direct = light_runtime.process_light(
            stimulus_light
        )

        rt = float(
            rgb["turn_raw_hz"]
        )

        dt = float(
            direct["turn_raw_hz"]
        )

        rm = float(
            rgb["movement_raw_hz"]
        )

        dm = float(
            direct["movement_raw_hz"]
        )

        rgb_turn.append(rt)
        direct_turn.append(dt)

        rgb_move.append(rm)
        direct_move.append(dm)

        turn_error = abs(
            rt - dt
        )

        move_error = abs(
            rm - dm
        )

        max_turn_error = max(
            max_turn_error,
            turn_error,
        )

        max_move_error = max(
            max_move_error,
            move_error,
        )

        if (
            step <= 5
            or step == STIMULUS_STEPS
        ):
            print(
                f"{step:>2} | "
                f"RGB turn={rt:+8.3f} "
                f"direct={dt:+8.3f} "
                f"err={turn_error:.6f} | "
                f"RGB move={rm:8.3f} "
                f"direct={dm:8.3f} "
                f"err={move_error:.6f}"
            )

    tail = slice(-7, None)

    return {
        "x": x_fraction,

        "rgb_turn": float(
            np.mean(
                rgb_turn[tail]
            )
        ),

        "direct_turn": float(
            np.mean(
                direct_turn[tail]
            )
        ),

        "rgb_move": float(
            np.mean(
                rgb_move[tail]
            )
        ),

        "direct_move": float(
            np.mean(
                direct_move[tail]
            )
        ),

        "max_turn_error": (
            max_turn_error
        ),

        "max_move_error": (
            max_move_error
        ),

        "retina_mean": float(
            stimulus_light.mean()
        ),
    }


def main():
    print()
    print(
        "FlyQ RGB vs Direct-Light Equivalence"
    )
    print(
        "===================================="
    )

    results = []

    for x in POSITIONS:
        results.append(
            run_position(x)
        )

    print()
    print("SUMMARY")
    print("=======")
    print()

    global_turn_error = 0.0
    global_move_error = 0.0

    for r in results:

        global_turn_error = max(
            global_turn_error,
            r["max_turn_error"],
        )

        global_move_error = max(
            global_move_error,
            r["max_move_error"],
        )

        print(
            f"x={r['x']:.2f} | "
            f"retina={r['retina_mean']:.4f} | "
            f"turn RGB={r['rgb_turn']:+7.2f} "
            f"direct={r['direct_turn']:+7.2f} | "
            f"move RGB={r['rgb_move']:7.2f} "
            f"direct={r['direct_move']:7.2f}"
        )

    print()
    print(
        f"GLOBAL MAX TURN ERROR : "
        f"{global_turn_error:.9f}"
    )

    print(
        f"GLOBAL MAX MOVE ERROR : "
        f"{global_move_error:.9f}"
    )

    if (
        global_turn_error < 1e-6
        and global_move_error < 1e-6
    ):
        print()
        print(
            "RESULT: EXACT MATCH ✅"
        )
        print(
            "RGB bridge is not causing "
            "the steering asymmetry."
        )

    else:
        print()
        print(
            "RESULT: PATHS DIFFER ⚠️"
        )
        print(
            "Inspect FlyQBrainRuntime."
        )


if __name__ == "__main__":
    main()
