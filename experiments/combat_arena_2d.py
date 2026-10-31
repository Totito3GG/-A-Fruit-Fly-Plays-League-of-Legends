from collections import deque
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np


# ============================================================
# Upstream DoomFly
# ============================================================

UPSTREAM = Path.home() / "FlyQ" / "upstream-doomfly"
sys.path.insert(0, str(UPSTREAM))

from doom.native import NativeBrain
from doom.engine import NeuralControls
from doom.game import retinal_samples

from control.attack_decoder import AttackDecoder


GRAPH = UPSTREAM / "outputs/doom/malecns_v1/graph.npz"
MANIFEST = UPSTREAM / "outputs/doom/malecns_v1/manifest.json"


# ============================================================
# Arena / renderer
# ============================================================

WIDTH = 320
HEIGHT = 240

FOV_DEG = 120.0
HALF_FOV_DEG = FOV_DEG / 2.0

TARGET_WIDTH = 64
TARGET_HEIGHT = 160

TARGET_DISTANCE = 0.80
TARGET_RADIUS = 0.12

TRIAL_BEARINGS = [
    +45.0,
    +25.0,
    +10.0,
    -10.0,
    -25.0,
    -45.0,
]


# ============================================================
# Frozen FlyQ navigation parameters
# ============================================================

STEP_MS = 28.6

TURN_STEP_DEG = 2.0
MOVE_STEP = 0.01

CENTER_HOLD_STEPS = 5

DRIVE_MOVE_STEPS = 6
DRIVE_MAX_NEURAL_STEPS = 20

RECOVERY_STEPS = 10

MAX_TRIAL_STEPS = 700


# Pixel steering frozen after previous experiments.

TURN_WINDOW = 7

LEFT_THRESHOLD = -3.0
RIGHT_THRESHOLD = +7.0

TURN_CONFIRM_STEPS = 2
TURN_COAST_STEPS = 5


# DNpe017 movement decoder frozen.

MOVE_ON_HZ = 30.0
MOVE_OFF_HZ = 25.0

MOVE_ON_CONFIRM = 3
MOVE_OFF_CONFIRM = 2


# Active visual search frozen.

RETINA_PRESENT_EPS = 1e-6

LOSS_CONFIRM_STEPS = 5
FOUND_CONFIRM_STEPS = 5

SEARCH_LIMIT_DEG = 80.0


# ============================================================
# Attack channel
# ============================================================

ATTACK_SUGAR_STEPS = 2

# Allows delayed MN9 onset after sugar stops.
ATTACK_EVENT_WAIT_STEPS = 10

# AttackDecoder itself requires five MN9-zero steps.
MAX_ATTACK_RECOVERY_STEPS = 300


# ============================================================
# Telemetry
# ============================================================

OUTPUT = Path(
    "telemetry/combat_arena_2d.csv"
)


# ============================================================
# Geometry
# ============================================================

def normalize_angle(angle):
    while angle > 180.0:
        angle -= 360.0

    while angle <= -180.0:
        angle += 360.0

    return angle


def target_xy(bearing_deg, distance):
    angle = math.radians(
        bearing_deg
    )

    return (
        distance * math.cos(angle),
        distance * math.sin(angle),
    )


def target_geometry(
    body_x,
    body_y,
    heading_deg,
    target_x,
    target_y,
):
    dx = target_x - body_x
    dy = target_y - body_y

    distance = math.hypot(
        dx,
        dy,
    )

    absolute_bearing = math.degrees(
        math.atan2(
            dy,
            dx,
        )
    )

    relative_bearing = normalize_angle(
        absolute_bearing
        - heading_deg
    )

    return (
        distance,
        relative_bearing,
    )


# ============================================================
# Rendering
# ============================================================

def black_frame():
    return np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )


def target_frame(
    relative_bearing_deg,
):
    """
    Renderer knows target bearing.

    Controller does NOT receive this value.

    Positive world bearing means target is
    visually left in our arena convention.
    """

    frame = black_frame()

    if abs(relative_bearing_deg) > HALF_FOV_DEG:
        return frame

    normalized = (
        relative_bearing_deg
        / HALF_FOV_DEG
    )

    center_x = int(
        round(
            WIDTH / 2
            - normalized
            * WIDTH / 2
        )
    )

    x0 = (
        center_x
        - TARGET_WIDTH // 2
    )

    x1 = (
        center_x
        + TARGET_WIDTH // 2
    )

    y0 = (
        HEIGHT // 2
        - TARGET_HEIGHT // 2
    )

    y1 = (
        HEIGHT // 2
        + TARGET_HEIGHT // 2
    )

    x0 = max(
        0,
        x0,
    )

    x1 = min(
        WIDTH,
        x1,
    )

    y0 = max(
        0,
        y0,
    )

    y1 = min(
        HEIGHT,
        y1,
    )

    if x1 > x0:
        frame[
            y0:y1,
            x0:x1,
        ] = 255

    return frame


def movement_cue_frame():
    """
    Same engineered bilateral peripheral
    movement cue used by FlyQ V2.
    """

    frame = black_frame()

    band = int(
        WIDTH * 0.30
    )

    frame[
        :,
        :band,
    ] = 188

    frame[
        :,
        WIDTH - band :,
    ] = 188

    return frame


# ============================================================
# Readout helpers
# ============================================================

def readout(
    readouts,
    neuron_type,
    side,
):
    for item in readouts:
        if (
            item.get("type")
            == neuron_type
            and
            item.get("side")
            == side
        ):
            return item

    raise RuntimeError(
        f"Missing "
        f"{neuron_type} "
        f"{side}"
    )


def indexes(
    readouts,
    neuron_type,
):
    return {
        item["side"]:
        int(item["index"])

        for item in readouts

        if (
            item.get("type")
            == neuron_type
        )
    }


# ============================================================
# Frozen steering decoder
# ============================================================

class PixelTurnDecoder:

    def __init__(
        self,
        baseline_hz=0.0,
    ):
        self.baseline_hz = float(
            baseline_hz
        )

        self.window = deque(
            maxlen=TURN_WINDOW
        )

    def reset_filter(self):
        self.window.clear()

    def update(
        self,
        raw_turn_hz,
    ):
        corrected = (
            float(raw_turn_hz)
            - self.baseline_hz
        )

        self.window.append(
            corrected
        )

        filtered = (
            sum(self.window)
            / len(self.window)
        )

        ready = (
            len(self.window)
            >= TURN_WINDOW
        )

        if not ready:
            direction = "CENTER"

        elif filtered < LEFT_THRESHOLD:
            direction = "LEFT"

        elif filtered > RIGHT_THRESHOLD:
            direction = "RIGHT"

        else:
            direction = "CENTER"

        return {
            "direction": direction,
            "corrected": corrected,
            "filtered": filtered,
            "ready": ready,
        }


# ============================================================
# Frozen persistent turning actuator
# ============================================================

class PersistentTurnActuator:

    def __init__(self):
        self.reset()

    def reset(self):
        self.pending = None
        self.pending_count = 0

        self.active = None
        self.coast_left = 0

    def update(
        self,
        direction,
    ):
        if direction in (
            "LEFT",
            "RIGHT",
        ):

            if self.pending == direction:
                self.pending_count += 1

            else:
                self.pending = direction
                self.pending_count = 1

            if (
                self.pending_count
                >= TURN_CONFIRM_STEPS
            ):
                self.active = direction

                self.coast_left = (
                    TURN_COAST_STEPS
                )

            return (
                self.active
                if self.active is not None
                else "CENTER"
            )

        self.pending = None
        self.pending_count = 0

        if (
            self.active is not None
            and
            self.coast_left > 0
        ):
            result = self.active

            self.coast_left -= 1

            if self.coast_left == 0:
                self.active = None

            return result

        self.active = None

        return "CENTER"


# ============================================================
# Frozen movement decoder
# ============================================================

class MovementBurstDecoder:

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "STOP"

        self.on_run = 0
        self.off_run = 0

    def update(
        self,
        movement_hz,
    ):
        movement_hz = float(
            movement_hz
        )

        if self.state == "STOP":

            self.off_run = 0

            if movement_hz >= MOVE_ON_HZ:
                self.on_run += 1
            else:
                self.on_run = 0

            if (
                self.on_run
                >= MOVE_ON_CONFIRM
            ):
                self.state = "MOVE"
                self.on_run = 0

        else:

            self.on_run = 0

            if movement_hz < MOVE_OFF_HZ:
                self.off_run += 1
            else:
                self.off_run = 0

            if (
                self.off_run
                >= MOVE_OFF_CONFIRM
            ):
                self.state = "STOP"
                self.off_run = 0

        return self.state


# ============================================================
# FlyQ persistent brain
# ============================================================

class CombatFlyQ:

    def __init__(self):

        self.manifest = json.loads(
            MANIFEST.read_text()
        )

        self.readouts = self.manifest[
            "readouts"
        ]

        self.brain = NativeBrain(
            GRAPH
        )

        self.controls = NeuralControls(
            self.readouts,
            mode="bci",
        )

        self.mn9_indexes = indexes(
            self.readouts,
            "MN9",
        )

        self.attack_decoder = (
            AttackDecoder(
                zero_confirm_steps=5
            )
        )

        self.turn_decoder = (
            PixelTurnDecoder()
        )

        self.turn_actuator = (
            PersistentTurnActuator()
        )

        self.movement_decoder = (
            MovementBurstDecoder()
        )

        self.neural_steps = 0

    def reset_navigation_filters(
        self,
    ):
        """
        Engineering state only.

        MaleCNS is NOT reset.
        """

        self.turn_decoder.reset_filter()
        self.turn_actuator.reset()
        self.movement_decoder.reset()

    def observe(
        self,
        frame,
        sugar=False,
    ):
        light = retinal_samples(
            frame,
            self.brain.uv,
        )

        counts, wall = self.brain.step(
            light,
            STEP_MS,
            sugar=sugar,
        )

        action = self.controls.decode(
            counts,
            STEP_MS / 1000.0,
        )

        readouts = action[
            "readouts"
        ]

        dnp_l = readout(
            readouts,
            "DNp20",
            "L",
        )

        dnp_r = readout(
            readouts,
            "DNp20",
            "R",
        )

        dnpe_l = readout(
            readouts,
            "DNpe017",
            "L",
        )

        dnpe_r = readout(
            readouts,
            "DNpe017",
            "R",
        )

        turn_raw = (
            float(
                dnp_r["rate_hz"]
            )
            -
            float(
                dnp_l["rate_hz"]
            )
        )

        movement_raw = (
            float(
                dnpe_l["rate_hz"]
            )
            +
            float(
                dnpe_r["rate_hz"]
            )
        )

        mn9_spikes = (
            int(
                counts[
                    self.mn9_indexes["L"]
                ]
            )
            +
            int(
                counts[
                    self.mn9_indexes["R"]
                ]
            )
        )

        attack_decision = (
            self.attack_decoder.update(
                mn9_spikes
            )
        )

        self.neural_steps += 1

        return {
            "retina_mean":
                float(
                    light.mean()
                ),

            "turn_raw_hz":
                turn_raw,

            "movement_raw_hz":
                movement_raw,

            "mn9_spikes":
                mn9_spikes,

            "attack_event":
                bool(
                    attack_decision.attack
                ),

            "attack_armed":
                bool(
                    attack_decision.armed
                ),

            "wall_seconds":
                float(wall),
        }

    def calibrate_black(
        self,
        steps=30,
    ):
        values = []

        frame = black_frame()

        for _ in range(steps):

            obs = self.observe(
                frame,
                sugar=False,
            )

            values.append(
                obs["turn_raw_hz"]
            )

        # Use settled tail.
        baseline = float(
            np.mean(
                values[-10:]
            )
        )

        self.turn_decoder.baseline_hz = (
            baseline
        )

        self.reset_navigation_filters()

        return baseline


# ============================================================
# Attack episode
# ============================================================

def perform_attack(
    fly,
):
    """
    Arena-controlled engineered attack trigger.

    True target distance determines whether this
    function is called.

    Sugar does NOT mean the simulated animal
    naturally understands combat.
    """

    frame = black_frame()

    events = 0

    pulse_observations = []

    # --------------------------------------------------------
    # 2-step engineered sugar pulse
    # --------------------------------------------------------

    for pulse_step in range(
        1,
        ATTACK_SUGAR_STEPS + 1,
    ):

        obs = fly.observe(
            frame,
            sugar=True,
        )

        pulse_observations.append(
            obs
        )

        if obs[
            "attack_event"
        ]:
            events += 1

        print(
            f"      sugar "
            f"{pulse_step}/"
            f"{ATTACK_SUGAR_STEPS}"
            f" | MN9="
            f"{obs['mn9_spikes']:>2}"
            f" | attack="
            f"{obs['attack_event']}"
        )

    # --------------------------------------------------------
    # Allow delayed MN9 onset
    # --------------------------------------------------------

    wait_steps = 0

    while (
        events == 0
        and
        wait_steps
        < ATTACK_EVENT_WAIT_STEPS
    ):

        wait_steps += 1

        obs = fly.observe(
            frame,
            sugar=False,
        )

        if obs[
            "attack_event"
        ]:
            events += 1

        print(
            f"      attack-wait "
            f"{wait_steps}"
            f" | MN9="
            f"{obs['mn9_spikes']:>2}"
            f" | attack="
            f"{obs['attack_event']}"
        )

    # --------------------------------------------------------
    # Allow MN9 latch to re-arm.
    # Same brain, no neural reset.
    # --------------------------------------------------------

    recovery_steps = 0

    while (
        not fly.attack_decoder.armed
        and
        recovery_steps
        < MAX_ATTACK_RECOVERY_STEPS
    ):

        recovery_steps += 1

        obs = fly.observe(
            frame,
            sugar=False,
        )

        if obs[
            "attack_event"
        ]:
            events += 1

        if (
            recovery_steps <= 5
            or
            recovery_steps % 20 == 0
            or
            fly.attack_decoder.armed
        ):

            print(
                f"      attack-recovery "
                f"{recovery_steps}"
                f" | MN9="
                f"{obs['mn9_spikes']:>2}"
                f" | armed="
                f"{fly.attack_decoder.armed}"
            )

    return {
        "events": events,
        "pulse_steps":
            ATTACK_SUGAR_STEPS,

        "wait_steps":
            wait_steps,

        "recovery_steps":
            recovery_steps,

        "rearmed":
            fly.attack_decoder.armed,
    }


# ============================================================
# Trial
# ============================================================

def run_trial(
    fly,
    trial_number,
    target_bearing,
    telemetry,
):

    target_x, target_y = (
        target_xy(
            target_bearing,
            TARGET_DISTANCE,
        )
    )

    body_x = 0.0
    body_y = 0.0
    heading = 0.0

    phase = "SEEK"

    center_hold = 0

    drive_moves = 0
    drive_neural_steps = 0

    recovery_step = 0

    missing_steps = 0

    search_found_steps = 0

    search_origin_heading = 0.0
    search_direction = +1

    accidental_attacks = 0

    target_attack_events = 0

    attack_requested = False
    target_defeated = False

    minimum_distance = (
        TARGET_DISTANCE
    )

    path_length = 0.0

    fly.reset_navigation_filters()

    print()
    print(
        "============================================"
    )

    print(
        f"TRIAL {trial_number} "
        f"| target bearing="
        f"{target_bearing:+.1f}°"
    )

    print(
        "============================================"
    )

    for step in range(
        1,
        MAX_TRIAL_STEPS + 1,
    ):

        (
            distance,
            relative_bearing,
        ) = target_geometry(
            body_x,
            body_y,
            heading,
            target_x,
            target_y,
        )

        minimum_distance = min(
            minimum_distance,
            distance,
        )

        # ====================================================
        # Arena attack-range trigger
        # ====================================================

        if (
            distance <= TARGET_RADIUS
            and
            not target_defeated
        ):

            attack_requested = True

            print()
            print(
                f">>> TARGET IN ENGINEERED "
                f"ATTACK RANGE"
            )

            print(
                f"    distance="
                f"{distance:.3f}"
            )

            attack_result = (
                perform_attack(
                    fly
                )
            )

            target_attack_events += (
                attack_result[
                    "events"
                ]
            )

            if (
                attack_result[
                    "events"
                ]
                == 1
            ):

                target_defeated = True

                print(
                    ">>> TARGET DEFEATED ✅"
                )

            else:

                print(
                    ">>> ATTACK CHANNEL "
                    "FAILED ❌"
                )

            break

        # ====================================================
        # SEEK
        # ====================================================

        if phase == "SEEK":

            frame = target_frame(
                relative_bearing
            )

            obs = fly.observe(
                frame,
                sugar=False,
            )

            if obs["attack_event"]:
                accidental_attacks += 1

            turn = (
                fly.turn_decoder.update(
                    obs[
                        "turn_raw_hz"
                    ]
                )
            )

            visible = (
                obs["retina_mean"]
                > RETINA_PRESENT_EPS
            )

            if visible:
                missing_steps = 0
            else:
                missing_steps += 1

            if (
                missing_steps
                >= LOSS_CONFIRM_STEPS
            ):

                phase = "SEARCH"

                search_origin_heading = (
                    heading
                )

                search_direction = +1

                search_found_steps = 0

                center_hold = 0

                fly.reset_navigation_filters()

                print(
                    f">>> SEARCH START "
                    f"at step {step}"
                )

                continue

            direction = (
                turn["direction"]
            )

            applied_turn = (
                fly.turn_actuator.update(
                    direction
                )
            )

            if applied_turn == "LEFT":

                heading = (
                    heading
                    + TURN_STEP_DEG
                )

            elif (
                applied_turn
                == "RIGHT"
            ):

                heading = (
                    heading
                    - TURN_STEP_DEG
                )

            heading = normalize_angle(
                heading
            )

            if (
                turn["ready"]
                and
                direction == "CENTER"
            ):

                center_hold += 1

            else:

                center_hold = 0

            if (
                center_hold
                >= CENTER_HOLD_STEPS
            ):

                phase = "DRIVE"

                drive_moves = 0
                drive_neural_steps = 0

                center_hold = 0

                fly.movement_decoder.reset()

        # ====================================================
        # SEARCH
        # ====================================================

        elif phase == "SEARCH":

            frame = target_frame(
                relative_bearing
            )

            obs = fly.observe(
                frame,
                sugar=False,
            )

            if obs["attack_event"]:
                accidental_attacks += 1

            visible = (
                obs["retina_mean"]
                > RETINA_PRESENT_EPS
            )

            if visible:
                search_found_steps += 1
            else:
                search_found_steps = 0

            if (
                search_found_steps
                >= FOUND_CONFIRM_STEPS
            ):

                phase = "SEEK"

                missing_steps = 0
                search_found_steps = 0

                fly.reset_navigation_filters()

                print(
                    f">>> TARGET REACQUIRED "
                    f"at step {step}"
                )

                continue

            if search_direction > 0:

                heading += (
                    TURN_STEP_DEG
                )

                swept = (
                    heading
                    - search_origin_heading
                )

                swept = normalize_angle(
                    swept
                )

                if (
                    swept
                    >= SEARCH_LIMIT_DEG
                ):
                    search_direction = -1

            else:

                heading -= (
                    TURN_STEP_DEG
                )

                swept = (
                    heading
                    - search_origin_heading
                )

                swept = normalize_angle(
                    swept
                )

                if (
                    swept
                    <= -SEARCH_LIMIT_DEG
                ):
                    search_direction = +1

            heading = normalize_angle(
                heading
            )

        # ====================================================
        # DRIVE
        # ====================================================

        elif phase == "DRIVE":

            frame = (
                movement_cue_frame()
            )

            obs = fly.observe(
                frame,
                sugar=False,
            )

            if obs["attack_event"]:
                accidental_attacks += 1

            move_state = (
                fly.movement_decoder.update(
                    obs[
                        "movement_raw_hz"
                    ]
                )
            )

            drive_neural_steps += 1

            if move_state == "MOVE":

                angle = math.radians(
                    heading
                )

                body_x += (
                    MOVE_STEP
                    * math.cos(angle)
                )

                body_y += (
                    MOVE_STEP
                    * math.sin(angle)
                )

                path_length += (
                    MOVE_STEP
                )

                drive_moves += 1

            if (
                drive_moves
                >= DRIVE_MOVE_STEPS
                or
                drive_neural_steps
                >= DRIVE_MAX_NEURAL_STEPS
            ):

                phase = "RECOVER"

                recovery_step = 0

        # ====================================================
        # RECOVER
        # ====================================================

        elif phase == "RECOVER":

            frame = target_frame(
                relative_bearing
            )

            obs = fly.observe(
                frame,
                sugar=False,
            )

            if obs["attack_event"]:
                accidental_attacks += 1

            recovery_step += 1

            if (
                recovery_step
                >= RECOVERY_STEPS
            ):

                phase = "SEEK"

                recovery_step = 0

                fly.reset_navigation_filters()

        else:
            raise RuntimeError(
                f"Unknown phase: {phase}"
            )

        # ====================================================
        # Telemetry
        # ====================================================

        (
            new_distance,
            new_relative,
        ) = target_geometry(
            body_x,
            body_y,
            heading,
            target_x,
            target_y,
        )

        minimum_distance = min(
            minimum_distance,
            new_distance,
        )

        telemetry.append({
            "trial":
                trial_number,

            "step":
                step,

            "phase":
                phase,

            "target_bearing_deg":
                target_bearing,

            "body_x":
                body_x,

            "body_y":
                body_y,

            "heading_deg":
                heading,

            "distance":
                new_distance,

            "relative_bearing_deg":
                new_relative,

            "minimum_distance":
                minimum_distance,

            "path_length":
                path_length,

            "accidental_attacks":
                accidental_attacks,

            "target_defeated":
                target_defeated,

            "total_neural_steps":
                fly.neural_steps,
        })

        if (
            step <= 10
            or
            step % 50 == 0
        ):

            print(
                f"{step:>4} "
                f"| {phase:<7} "
                f"| d="
                f"{new_distance:.3f} "
                f"| rel="
                f"{new_relative:+6.1f}° "
                f"| heading="
                f"{heading:+6.1f}°"
            )

    final_distance, _ = (
        target_geometry(
            body_x,
            body_y,
            heading,
            target_x,
            target_y,
        )
    )

    success = (
        target_defeated
        and
        target_attack_events == 1
        and
        accidental_attacks == 0
    )

    return {
        "trial":
            trial_number,

        "bearing":
            target_bearing,

        "success":
            success,

        "target_defeated":
            target_defeated,

        "attack_requested":
            attack_requested,

        "target_attack_events":
            target_attack_events,

        "accidental_attacks":
            accidental_attacks,

        "min_distance":
            minimum_distance,

        "final_distance":
            final_distance,

        "path_length":
            path_length,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "FlyQ 35A — Three-Action Combat Arena"
    )

    print(
        "===================================="
    )

    print()

    print(
        "TURN   : DNp20 R-L"
    )

    print(
        "MOVE   : DNpe017 L+R"
    )

    print(
        "ATTACK : MN9 onset + engineered latch"
    )

    print()

    print(
        "Attack trigger:"
    )

    print(
        "Arena supplies 2-step sugar cue only"
    )

    print(
        "when true target distance <= 0.12."
    )

    print()

    print(
        "This is an engineered combat interface,"
    )

    print(
        "not a claim of natural fly attack behavior."
    )

    print()

    fly = CombatFlyQ()

    print(
        "Calibrating black background..."
    )

    baseline = (
        fly.calibrate_black(
            steps=30
        )
    )

    print(
        f"Black DNp20 baseline: "
        f"{baseline:+.3f} Hz"
    )

    print()

    telemetry = []
    results = []

    neural_start = (
        fly.neural_steps
    )

    for trial_number, bearing in enumerate(
        TRIAL_BEARINGS,
        start=1,
    ):

        result = run_trial(
            fly,
            trial_number,
            bearing,
            telemetry,
        )

        results.append(
            result
        )

        print()

        print(
            f"TRIAL {trial_number} RESULT: "
            f"{'PASS ✅' if result['success'] else 'FAIL ❌'}"
        )

        print(
            f"  minimum distance: "
            f"{result['min_distance']:.3f}"
        )

        print(
            f"  target attack events: "
            f"{result['target_attack_events']}"
        )

        print(
            f"  accidental attacks: "
            f"{result['accidental_attacks']}"
        )

    # ========================================================
    # Telemetry
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if telemetry:

        with OUTPUT.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=(
                    telemetry[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                telemetry
            )

    # ========================================================
    # Final summary
    # ========================================================

    successes = sum(
        1
        for result in results
        if result["success"]
    )

    defeated = sum(
        1
        for result in results
        if result[
            "target_defeated"
        ]
    )

    target_attacks = sum(
        result[
            "target_attack_events"
        ]
        for result in results
    )

    accidental = sum(
        result[
            "accidental_attacks"
        ]
        for result in results
    )

    print()
    print(
        "===================================="
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "===================================="
    )

    print()

    for result in results:

        print(
            f"Trial "
            f"{result['trial']} "
            f"{result['bearing']:+5.1f}° "
            f"| min_d="
            f"{result['min_distance']:.3f} "
            f"| attack="
            f"{result['target_attack_events']} "
            f"| accidental="
            f"{result['accidental_attacks']} "
            f"| "
            f"{'✅' if result['success'] else '❌'}"
        )

    print()

    print(
        f"Combat successes: "
        f"{successes}/"
        f"{len(TRIAL_BEARINGS)}"
    )

    print(
        f"Targets defeated: "
        f"{defeated}/"
        f"{len(TRIAL_BEARINGS)}"
    )

    print(
        f"Target attack events: "
        f"{target_attacks}"
    )

    print(
        f"Accidental attacks: "
        f"{accidental}"
    )

    print(
        f"Experiment neural steps: "
        f"{fly.neural_steps - neural_start}"
    )

    print(
        "MaleCNS instances: 1"
    )

    print(
        "Neural resets: 0"
    )

    print()

    full_pass = (
        successes
        == len(
            TRIAL_BEARINGS
        )
        and
        defeated
        == len(
            TRIAL_BEARINGS
        )
        and
        target_attacks
        == len(
            TRIAL_BEARINGS
        )
        and
        accidental == 0
    )

    if full_pass:

        print(
            "35A THREE-ACTION COMBAT: "
            "PASS ✅"
        )

    else:

        print(
            "35A THREE-ACTION COMBAT: "
            "NOT YET PASSED ❌"
        )

    print()

    print(
        f"Telemetry saved → "
        f"{OUTPUT}"
    )


if __name__ == "__main__":
    main()
