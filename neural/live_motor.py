import time

from neural.client import FlyQNeuralClient
from neural.readouts import extract_motor_signals


def main():
    client = FlyQNeuralClient()

    print("FlyQ Neural Motor Monitor")
    print("=========================")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            state = client.state()
            motor = extract_motor_signals(state)

            print(
                f"seq={state['sequence']:>6} | "
                f"DNp20 L={motor.dnp20_left_hz:>7.2f} "
                f"R={motor.dnp20_right_hz:>7.2f} | "
                f"turn={motor.turn_signal:>8.2f} | "
                f"DNpe017 L={motor.dnpe017_left_hz:>7.2f} "
                f"R={motor.dnpe017_right_hz:>7.2f} | "
                f"move={motor.movement_signal:>8.2f}"
            )

            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\nFlyQ motor monitor stopped.")


if __name__ == "__main__":
    main()
