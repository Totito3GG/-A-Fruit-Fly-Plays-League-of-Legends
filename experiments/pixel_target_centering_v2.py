import csv
import gc
from pathlib import Path

from experiments.pixel_target_centering import (
    PixelController,
    render_target,
    wrap_angle,
    TARGET_BEARINGS,
    TARGET_ANGLE_TOLERANCE,
    STABLE_STEPS,
    MAX_STEPS,
    TURN_STEP_DEG,
)


OUTPUT = Path(
    "telemetry/pixel_target_centering_v2.csv"
)

#
# Require the neural decoder to give the same
# non-center direction twice before accepting it.
#
CONFIRM_STEPS = 2

#
# Once a direction has been confirmed, preserve
# that motor command for this many neural steps
# through temporary CENTER evidence.
#
# 5 × 2° = at most 10° of additional rotation.
#
COAST_STEPS = 5


class PersistentTurnActuator:
    """
    Converts neural direction evidence into body rotation.

    It receives only:
        LEFT / CENTER / RIGHT

    It never sees:
        target position
        target angle
        image x coordinate
        evaluation error
    """

    def __init__(self):
        self.streak_direction = None
        self.streak_count = 0

        self.last_confirmed = "CENTER"
        self.coast_remaining = 0

    def update(self, evidence):
        #
        # Strong directional evidence.
        #
        if evidence in {
            "LEFT",
            "RIGHT",
        }:
            if (
                evidence
                == self.streak_direction
            ):
                self.streak_count += 1
            else:
                self.streak_direction = (
                    evidence
                )
                self.streak_count = 1

            #
            # Two consecutive neural decisions
            # confirm the motor direction.
            #
            if (
                self.streak_count
                >= CONFIRM_STEPS
            ):
                self.last_confirmed = (
                    evidence
                )

                self.coast_remaining = (
                    COAST_STEPS
                )

                return (
                    evidence,
                    "NEURAL",
                )

        else:
            #
            # CENTER breaks directional
            # confirmation streak.
            #
            self.streak_direction = None
            self.streak_count = 0

        #
        # Weak/ambiguous period:
        # briefly preserve the last confirmed
        # neural motor command.
        #
        if (
            self.last_confirmed
            in {"LEFT", "RIGHT"}
            and self.coast_remaining > 0
        ):
            self.coast_remaining -= 1

            return (
                self.last_confirmed,
                "COAST",
            )

        self.last_confirmed = "CENTER"

        return (
            "CENTER",
            "STOP",
        )


def run_trial(
    trial_number,
    target_bearing,
):
    controller = PixelController()

    baseline = (
        controller.calibrate_black()
    )

    actuator = (
        PersistentTurnActuator()
    )

    heading = 0.0
    stable = 0

    telemetry = []

    acquired = False

    previous_action = None

    print()
    print(
        f"===== TRIAL {trial_number} ====="
    )

    print(
        f"Target bearing : "
        f"{target_bearing:+.1f}°"
    )

    print(
        f"Black baseline : "
        f"{baseline:+.2f} Hz"
    )

    print()

    for step in range(
        1,
        MAX_STEPS + 1,
    ):
        #
        # WORLD / RENDERER
        #
        relative_angle = wrap_angle(
            target_bearing
            - heading
        )

        (
            frame,
            target_x,
        ) = render_target(
            relative_angle
        )

        #
        # BRAIN
        #
        # Controller receives RGB pixels only.
        #
        neural = controller.step(
            frame
        )

        evidence = neural[
            "direction"
        ]

        #
        # MOTOR INTERFACE
        #
        action, action_source = (
            actuator.update(
                evidence
            )
        )

        if neural["ready"]:
            if action == "LEFT":
                heading = wrap_angle(
                    heading
                    + TURN_STEP_DEG
                )

            elif action == "RIGHT":
                heading = wrap_angle(
                    heading
                    - TURN_STEP_DEG
                )

        #
        # EVALUATION ONLY.
        # Never sent to controller or actuator.
        #
        error = wrap_angle(
            target_bearing
            - heading
        )

        centered = (
            abs(error)
            <= TARGET_ANGLE_TOLERANCE
        )

        #
        # Success requires:
        # - physically within tolerance
        # - actuator no longer turning
        #
        if (
            centered
            and neural["ready"]
            and action == "CENTER"
        ):
            stable += 1
        else:
            stable = 0

        telemetry.append({
            "trial": trial_number,
            "step": step,
            "target_bearing_deg": (
                target_bearing
            ),
            "heading_deg": heading,
            "error_deg": error,

            "target_x": (
                ""
                if target_x is None
                else target_x
            ),

            "neural_evidence": evidence,
            "motor_action": action,
            "action_source": (
                action_source
            ),

            "raw_turn_hz": (
                neural[
                    "raw_turn_hz"
                ]
            ),

            "turn_delta_hz": (
                neural[
                    "turn_delta_hz"
                ]
            ),

            "filtered_delta_hz": (
                neural[
                    "filtered_delta_hz"
                ]
            ),

            "stable": stable,
        })

        should_print = (
            step == 1
            or step % 10 == 0
            or action
            != previous_action
            or abs(error) <= 12.0
        )

        if should_print:
            x_text = (
                "OFF"
                if target_x is None
                else f"{target_x:.3f}"
            )

            print(
                f"{step:>3} | "
                f"x={x_text:>5} | "
                f"brain={evidence:>6} | "
                f"motor={action:>6} | "
                f"src={action_source:>6} | "
                f"filtΔ="
                f"{neural['filtered_delta_hz']:+7.2f} | "
                f"head={heading:+7.1f}° | "
                f"err={error:+7.1f}° | "
                f"stable={stable}"
            )

        previous_action = action

        if stable >= STABLE_STEPS:
            acquired = True

            print()
            print(
                "TARGET CENTERED ✅ "
                f"after {step} neural steps"
            )

            print(
                f"Final heading : "
                f"{heading:+.1f}°"
            )

            print(
                f"Final error   : "
                f"{error:+.1f}°"
            )

            break

    if not acquired:
        print()
        print(
            "TARGET NOT CENTERED ❌"
        )

        print(
            f"Final heading : "
            f"{heading:+.1f}°"
        )

        print(
            f"Final error   : "
            f"{error:+.1f}°"
        )

    del controller
    gc.collect()

    return {
        "success": acquired,
        "heading": heading,
        "error": error,
        "steps": len(
            telemetry
        ),
        "telemetry": telemetry,
    }


def main():
    print()
    print(
        "FlyQ Pixel Target Centering V2"
    )
    print(
        "=============================="
    )

    print()
    print(
        "RGB pixels -> MaleCNS -> "
        "DNp20 -> persistent actuator"
    )

    print()

    print(
        f"Confirmation : "
        f"{CONFIRM_STEPS} neural decisions"
    )

    print(
        f"Motor coast  : "
        f"{COAST_STEPS} steps"
    )

    print(
        f"Turn step    : "
        f"{TURN_STEP_DEG:.1f}°"
    )

    print()

    all_rows = []
    summary = []

    successes = 0

    for trial, bearing in enumerate(
        TARGET_BEARINGS,
        start=1,
    ):
        result = run_trial(
            trial,
            bearing,
        )

        successes += int(
            result["success"]
        )

        all_rows.extend(
            result["telemetry"]
        )

        summary.append((
            bearing,
            result,
        ))

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_rows[
                0
            ].keys(),
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    print()
    print(
        "=============================="
    )
    print(
        "FINAL SUMMARY"
    )
    print(
        "=============================="
    )
    print()

    for i, (
        bearing,
        result,
    ) in enumerate(
        summary,
        start=1,
    ):
        status = (
            "✅"
            if result["success"]
            else "❌"
        )

        print(
            f"Trial {i} | "
            f"target={bearing:+6.1f}° | "
            f"steps={result['steps']:>3} | "
            f"final="
            f"{result['heading']:+6.1f}° | "
            f"error="
            f"{result['error']:+6.1f}° | "
            f"{status}"
        )

    print()

    print(
        f"Targets centered: "
        f"{successes}/"
        f"{len(TARGET_BEARINGS)}"
    )

    print()

    print(
        f"Telemetry saved → "
        f"{OUTPUT}"
    )


if __name__ == "__main__":
    main()
