from collections import deque
from pathlib import Path
from statistics import mean, pstdev
import json
import time

from neural.client import FlyQNeuralClient
from neural.readouts import extract_motor_signals


TARGET_SAMPLES = 300
WINDOW_SIZE = 7

OUTPUT = Path("telemetry/live_calibration.json")


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def stats(values):
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "std": pstdev(values),
        "p05": percentile(values, 0.05),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
    }


def print_stats(name, result):
    print()
    print(name)
    print("-" * len(name))

    for key, value in result.items():
        if key == "count":
            print(f"{key:>6}: {value}")
        else:
            print(f"{key:>6}: {value:8.3f} Hz")


def main():
    client = FlyQNeuralClient()

    history = deque(maxlen=WINDOW_SIZE)

    raw_values = []
    filtered_values = []

    last_sequence = None
    run_id = None

    print()
    print("FlyQ Live Neural Calibration")
    print("============================")
    print(f"Collecting {TARGET_SAMPLES} UNIQUE neural states")
    print()

    while len(raw_values) < TARGET_SAMPLES:
        state = client.state()

        current_run = state["run_id"]
        sequence = state["sequence"]

        if run_id is None:
            run_id = current_run

        if current_run != run_id:
            print("Server restarted — restarting calibration.")

            run_id = current_run
            last_sequence = None
            history.clear()
            raw_values.clear()
            filtered_values.clear()

        if sequence == last_sequence:
            time.sleep(0.01)
            continue

        last_sequence = sequence

        motor = extract_motor_signals(state)

        raw = float(motor.turn_signal)

        raw_values.append(raw)

        history.append(raw)

        if len(history) == WINDOW_SIZE:
            filtered_values.append(
                sum(history) / WINDOW_SIZE
            )

        if len(raw_values) % 50 == 0:
            print(
                f"{len(raw_values):>3}/{TARGET_SAMPLES} "
                f"unique states collected"
            )

    raw_stats = stats(raw_values)
    filtered_stats = stats(filtered_values)

    print_stats(
        "RAW DNp20 R-L",
        raw_stats,
    )

    print_stats(
        "7-STATE FILTERED DNp20 R-L",
        filtered_stats,
    )

    left_old = sum(
        x < 0.0
        for x in filtered_values
    )

    right_old = sum(
        x > 17.0
        for x in filtered_values
    )

    center_old = (
        len(filtered_values)
        - left_old
        - right_old
    )

    print()
    print("CURRENT DECODER")
    print("===============")

    print(
        f"LEFT   (< 0)  : "
        f"{left_old}"
    )

    print(
        f"CENTER        : "
        f"{center_old}"
    )

    print(
        f"RIGHT  (>17)  : "
        f"{right_old}"
    )

    result = {
        "run_id": run_id,
        "unique_samples": len(raw_values),
        "window_size": WINDOW_SIZE,
        "raw": raw_stats,
        "filtered": filtered_stats,
        "current_decoder": {
            "left_threshold": 0.0,
            "right_threshold": 17.0,
            "left_count": left_old,
            "center_count": center_old,
            "right_count": right_old,
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        f"Saved → {OUTPUT}"
    )


if __name__ == "__main__":
    main()
