import time

from neural.client import FlyQNeuralClient
from neural.readouts import extract_motor_signals
from control.direction_decoder import DirectionDecoder


def main():
    client = FlyQNeuralClient()

    decoder = DirectionDecoder(
        window_size=7,
        left_threshold=0.0,
        right_threshold=17.0,
    )

    last_sequence = None
    last_run_id = None

    print()
    print("FlyQ Live Direction Decoder")
    print("===========================")
    print()
    print("LEFT   < 0 Hz")
    print("CENTER = 0–17 Hz")
    print("RIGHT  > 17 Hz")
    print("Window = 7 UNIQUE neural states (~200 ms simulated)")
    print()

    try:
        while True:
            state = client.state()

            sequence = state["sequence"]
            run_id = state["run_id"]

            # If DoomFly was restarted, throw away old history.
            if run_id != last_run_id:
                decoder.reset()
                last_sequence = None
                last_run_id = run_id

                print(
                    f"[new neural run: {run_id}]"
                )

            # Critical:
            # never feed the same neural state into the decoder twice.
            if sequence == last_sequence:
                time.sleep(0.01)
                continue

            last_sequence = sequence

            motor = extract_motor_signals(
                state
            )

            decision = decoder.update(
                motor.turn_signal
            )

            print(
                f"seq={sequence:>6} | "
                f"raw={decision.raw_turn_hz:>7.2f} Hz | "
                f"avg={decision.filtered_turn_hz:>7.2f} Hz | "
                f"{decision.direction}"
            )

            time.sleep(0.01)

    except KeyboardInterrupt:
        print(
            "\nFlyQ direction decoder stopped."
        )


if __name__ == "__main__":
    main()
