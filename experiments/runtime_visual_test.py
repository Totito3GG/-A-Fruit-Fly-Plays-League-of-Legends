import numpy as np

from neural.runtime import FlyQBrainRuntime


WIDTH = 320
HEIGHT = 240

STEPS_PER_CONDITION = 14


def frame(condition):
    image = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )

    if condition == "black":
        return image

    if condition == "left":
        image[:, : WIDTH // 2] = 255

    elif condition == "center":
        start = WIDTH // 3
        end = 2 * WIDTH // 3

        image[:, start:end] = 255

    elif condition == "right":
        image[:, WIDTH // 2 :] = 255

    elif condition == "white":
        image[:] = 255

    else:
        raise ValueError(condition)

    return image


def main():
    runtime = FlyQBrainRuntime()

    print()
    print("FlyQ Standalone Visual Runtime")
    print("==============================")
    print()

    conditions = [
        "black",
        "left",
        "black",
        "center",
        "black",
        "right",
        "black",
    ]

    for condition in conditions:

        print()
        print(
            f"===== {condition.upper()} ====="
        )

        image = frame(condition)

        for _ in range(STEPS_PER_CONDITION):

            result = runtime.process_frame(
                image
            )

            print(
                f"seq={result['sequence']:>3} | "
                f"retina={result['retina_mean']:.3f} | "
                f"raw={result['turn_raw_hz']:>7.2f} | "
                f"avg={result['filtered_turn_hz']:>7.2f} | "
                f"{result['direction']}"
            )


if __name__ == "__main__":
    main()
