import json
import time
from urllib.request import urlopen
from urllib.error import URLError


class FlyQNeuralClient:
    def __init__(self, host="127.0.0.1", port=18765):
        self.base_url = f"http://{host}:{port}"

    def _get_json(self, endpoint):
        with urlopen(f"{self.base_url}{endpoint}", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self):
        return self._get_json("/health")

    def state(self):
        return self._get_json("/state")


def main():
    client = FlyQNeuralClient()

    print("FlyQ neural bridge")
    print("==================")

    try:
        health = client.health()

        print(f"Status : {health.get('status')}")
        print(f"Run ID : {health.get('run_id')}")
        print(f"Seq    : {health.get('sequence')}")
        print()
        print("Watching neural server... Ctrl+C to stop.")
        print()

        while True:
            state = client.state()

            # Deliberately do NOT print the Base64 video frame.
            clean_state = {
                key: value
                for key, value in state.items()
                if key != "frame"
            }

            print(
                f"seq={clean_state.get('sequence')} "
                f"status={clean_state.get('status')} "
                f"condition={clean_state.get('condition')} "
                f"decoder={clean_state.get('decoder')}"
            )

            time.sleep(1)

    except URLError as exc:
        print(f"Cannot reach FlyQ neural server: {exc}")
        print("Make sure DoomFly is running on port 18765.")

    except KeyboardInterrupt:
        print("\nFlyQ client stopped.")


if __name__ == "__main__":
    main()
