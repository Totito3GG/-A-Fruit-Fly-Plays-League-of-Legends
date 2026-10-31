from dataclasses import dataclass
from typing import Any


@dataclass
class MotorSignals:
    dnp20_left_hz: float
    dnp20_right_hz: float

    dnpe017_left_hz: float
    dnpe017_right_hz: float

    turn_signal: float
    movement_signal: float

    dnp20_left_spikes: int
    dnp20_right_spikes: int
    dnpe017_left_spikes: int
    dnpe017_right_spikes: int


def _find_readout(
    readouts: list[dict[str, Any]],
    neuron_type: str,
    side: str,
) -> dict[str, Any]:
    for readout in readouts:
        if (
            readout.get("type") == neuron_type
            and readout.get("side") == side
        ):
            return readout

    raise RuntimeError(
        f"Missing neural readout: {neuron_type} side={side}"
    )


def extract_motor_signals(state: dict[str, Any]) -> MotorSignals:
    readouts = state.get("readouts", [])

    dnp20_left = _find_readout(readouts, "DNp20", "L")
    dnp20_right = _find_readout(readouts, "DNp20", "R")

    dnpe017_left = _find_readout(readouts, "DNpe017", "L")
    dnpe017_right = _find_readout(readouts, "DNpe017", "R")

    left_turn = float(dnp20_left["rate_hz"])
    right_turn = float(dnp20_right["rate_hz"])

    left_move = float(dnpe017_left["rate_hz"])
    right_move = float(dnpe017_right["rate_hz"])

    return MotorSignals(
        dnp20_left_hz=left_turn,
        dnp20_right_hz=right_turn,

        dnpe017_left_hz=left_move,
        dnpe017_right_hz=right_move,

        # Raw FlyQ signals — not yet League commands.
        turn_signal=right_turn - left_turn,
        movement_signal=left_move + right_move,

        dnp20_left_spikes=int(dnp20_left["spikes"]),
        dnp20_right_spikes=int(dnp20_right["spikes"]),
        dnpe017_left_spikes=int(dnpe017_left["spikes"]),
        dnpe017_right_spikes=int(dnpe017_right["spikes"]),
    )
