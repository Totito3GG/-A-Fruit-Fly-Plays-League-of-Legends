from pathlib import Path
import json
import sys

import numpy as np


UPSTREAM = Path.home() / "FlyQ" / "upstream-doomfly"
sys.path.insert(0, str(UPSTREAM))

from doom.native import NativeBrain
from doom.game import retinal_samples
from doom.engine import NeuralControls

from control.direction_decoder import DirectionDecoder
from neural.readouts import extract_motor_signals


GRAPH = UPSTREAM / "outputs/doom/malecns_v1/graph.npz"
MANIFEST = UPSTREAM / "outputs/doom/malecns_v1/manifest.json"


class FlyQBrainRuntime:
    """
    Persistent FlyQ MaleCNS runtime.

    One runtime owns one continuously evolving simulated brain.
    """

    def __init__(
        self,
        step_ms: float = 28.6,
        window_size: int = 7,
        left_threshold: float = 0.0,
        right_threshold: float = 17.0,
    ):
        self.step_ms = step_ms

        manifest = json.loads(
            MANIFEST.read_text()
        )

        self.brain = NativeBrain(GRAPH)

        self.controls = NeuralControls(
            manifest["readouts"],
            mode="bci",
        )

        self.decoder = DirectionDecoder(
            window_size=window_size,
            left_threshold=left_threshold,
            right_threshold=right_threshold,
        )

        self.sequence = 0

    def process_frame(
        self,
        frame: np.ndarray,
    ):
        """
        RGB image -> retina -> MaleCNS -> DNp20 -> FlyQ direction.
        """

        light = retinal_samples(
            frame,
            self.brain.uv,
        )

        return self.process_light(light)

    def process_light(
        self,
        light: np.ndarray,
    ):
        """
        Direct retinal luminance input.

        Useful for controlled experiments.
        """

        light = np.asarray(
            light,
            dtype=np.float32,
        )

        if len(light) != len(self.brain.uv):
            raise ValueError(
                f"Expected {len(self.brain.uv)} retinal inputs, "
                f"got {len(light)}"
            )

        counts, wall = self.brain.step(
            light,
            self.step_ms,
        )

        action = self.controls.decode(
            counts,
            self.step_ms / 1000,
        )

        motor = extract_motor_signals({
            "readouts": action["readouts"]
        })

        decision = self.decoder.update(
            motor.turn_signal
        )

        self.sequence += 1

        return {
            "sequence": self.sequence,

            "retina_mean": float(
                light.mean()
            ),

            "turn_raw_hz": float(
                motor.turn_signal
            ),

            "movement_raw_hz": float(
                motor.movement_signal
            ),

            "direction": decision.direction,

            "filtered_turn_hz": float(
                decision.filtered_turn_hz
            ),

            "ready": bool(
                decision.ready
            ),

            "doom_action": {
                "turn": float(
                    action["turn"]
                ),
                "forward": float(
                    action["forward"]
                ),
                "attack": bool(
                    action["attack"]
                ),
            },

            "wall_seconds": float(
                wall
            ),
        }
