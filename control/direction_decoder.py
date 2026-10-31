from collections import deque
from dataclasses import dataclass


@dataclass
class DirectionDecision:
    direction: str
    raw_turn_hz: float
    filtered_turn_hz: float
    samples: int
    ready: bool


class DirectionDecoder:
    """
    First FlyQ lateral-action decoder.

    Calibration comes from the controlled MaleCNS experiments.

    The neural signal is:

        DNp20_R rate - DNp20_L rate

    We smooth several neural windows because the temporal probe showed
    that the first ~50 ms is not direction-specific and useful separation
    develops over roughly 150-200 ms.
    """

    def __init__(
        self,
        window_size: int = 7,
        left_threshold: float = 0.0,
        right_threshold: float = 17.0,
    ):
        self.window_size = window_size
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold

        self._history = deque(
            maxlen=window_size
        )

    def reset(self):
        self._history.clear()

    def update(
        self,
        turn_signal_hz: float,
    ) -> DirectionDecision:

        self._history.append(
            float(turn_signal_hz)
        )

        filtered = (
            sum(self._history)
            / len(self._history)
        )

        ready = (
            len(self._history)
            == self.window_size
        )

        if not ready:
            direction = "WARMUP"

        elif filtered < self.left_threshold:
            direction = "LEFT"

        elif filtered > self.right_threshold:
            direction = "RIGHT"

        else:
            direction = "CENTER"

        return DirectionDecision(
            direction=direction,
            raw_turn_hz=float(
                turn_signal_hz
            ),
            filtered_turn_hz=filtered,
            samples=len(self._history),
            ready=ready,
        )
