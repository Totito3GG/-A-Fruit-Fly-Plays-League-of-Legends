from dataclasses import dataclass


@dataclass
class AttackDecision:
    attack: bool
    armed: bool
    mn9_spikes: int
    zero_run: int
    event_count: int


class AttackDecoder:
    """
    FlyQ engineered MN9 attack-event decoder.

    Important:
    - MN9 is not interpreted as a natural "attack neuron".
    - MN9 activity is currently evoked through DoomFly's engineered
      sugar sensory drive.
    - One MN9 activation episode produces ONE attack event.
    - Persistent MN9 firing is ignored until activity has been absent
      for several consecutive neural windows.
    """

    def __init__(
        self,
        zero_confirm_steps: int = 5,
    ):
        if zero_confirm_steps < 1:
            raise ValueError(
                "zero_confirm_steps must be >= 1"
            )

        self.zero_confirm_steps = (
            zero_confirm_steps
        )

        self.armed = True
        self.zero_run = 0
        self.event_count = 0

    def reset_engineering_state(self) -> None:
        """
        Reset decoder/latch state only.

        This does NOT reset MaleCNS neural state.
        """
        self.armed = True
        self.zero_run = 0
        self.event_count = 0

    def update(
        self,
        mn9_spikes: int,
    ) -> AttackDecision:

        mn9_spikes = int(
            mn9_spikes
        )

        if mn9_spikes < 0:
            raise ValueError(
                "mn9_spikes cannot be negative"
            )

        attack = False

        if mn9_spikes > 0:

            self.zero_run = 0

            if self.armed:

                attack = True

                self.event_count += 1

                self.armed = False

        else:

            self.zero_run += 1

            if (
                not self.armed
                and
                self.zero_run
                >= self.zero_confirm_steps
            ):

                self.armed = True

        return AttackDecision(
            attack=attack,
            armed=self.armed,
            mn9_spikes=mn9_spikes,
            zero_run=self.zero_run,
            event_count=self.event_count,
        )
