from dataclasses import dataclass


@dataclass
class MovementDecision:
    state: str
    movement_hz: float
    on_count: int
    off_count: int


class MovementDecoder:
    """
    FlyQ DNpe017 movement decoder.

    Neural signal:
        DNpe017_L + DNpe017_R

    Calibration:
        baseline / stop-like activity ~17-21 Hz
        bilateral_outer activity     ~48-50 Hz

    Hysteresis prevents brief threshold rebounds
    from causing MOVE/STOP flicker.
    """

    def __init__(
        self,
        on_threshold: float = 30.0,
        off_threshold: float = 25.0,
        on_samples: int = 3,
        off_samples: int = 2,
    ):
        if off_threshold >= on_threshold:
            raise ValueError(
                "off_threshold must be lower than on_threshold"
            )

        self.on_threshold = on_threshold
        self.off_threshold = off_threshold

        self.on_samples = on_samples
        self.off_samples = off_samples

        self.state = "STOP"

        self._on_count = 0
        self._off_count = 0

    def reset(self):
        self.state = "STOP"
        self._on_count = 0
        self._off_count = 0

    def update(
        self,
        movement_hz: float,
    ) -> MovementDecision:

        movement_hz = float(movement_hz)

        if self.state == "STOP":

            self._off_count = 0

            if movement_hz >= self.on_threshold:
                self._on_count += 1
            else:
                self._on_count = 0

            if self._on_count >= self.on_samples:
                self.state = "MOVE"
                self._on_count = 0

        else:

            self._on_count = 0

            if movement_hz < self.off_threshold:
                self._off_count += 1
            else:
                self._off_count = 0

            if self._off_count >= self.off_samples:
                self.state = "STOP"
                self._off_count = 0

        return MovementDecision(
            state=self.state,
            movement_hz=movement_hz,
            on_count=self._on_count,
            off_count=self._off_count,
        )
