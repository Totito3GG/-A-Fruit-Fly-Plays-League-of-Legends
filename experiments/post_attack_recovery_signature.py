from collections import deque
import numpy as np

from experiments.combat_arena_2d import (
    CombatFlyQ,
    black_frame,
    run_trial,
    STEP_MS,
)


BASELINE_STEPS = 50

ROLLING_WINDOW = 20

MAX_POST_ATTACK_STEPS = 2000

# After first matching window, require the condition
# to remain valid for this many consecutive windows.
STABLE_WINDOWS_REQUIRED = 20


def mean_std(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(values.mean()),
        float(values.std()),
    )


def tolerance(mean, std, relative_floor=0.15):
    """
    Engineering tolerance.

    Prefer baseline variability (3 sigma),
    with a small relative floor when the
    deterministic baseline has tiny variance.
    """

    return max(
        3.0 * std,
        relative_floor
        * max(abs(mean), 1.0),
    )


def sample(fly):
    obs = fly.observe(
        black_frame(),
        sugar=False,
    )

    total_spikes = int(
        fly.brain.counts.sum()
    )

    return {
        "turn": float(
            obs["turn_raw_hz"]
        ),

        "move": float(
            obs["movement_raw_hz"]
        ),

        "mn9": int(
            obs["mn9_spikes"]
        ),

        "attack_event": bool(
            obs["attack_event"]
        ),

        "total_spikes": total_spikes,
    }


def main():

    print()
    print(
        "FlyQ Post-Attack Recovery Signature"
    )
    print(
        "=================================="
    )
    print()

    fly = CombatFlyQ()

    print(
        "Initial black calibration..."
    )

    baseline_turn = (
        fly.calibrate_black(
            steps=30
        )
    )

    print(
        f"DNp20 calibration baseline: "
        f"{baseline_turn:+.3f} Hz"
    )

    print()
    print(
        f"Collecting {BASELINE_STEPS} "
        f"pre-combat black steps..."
    )

    baseline = []

    for i in range(
        1,
        BASELINE_STEPS + 1,
    ):

        row = sample(
            fly
        )

        baseline.append(
            row
        )

        if (
            i <= 5
            or i % 10 == 0
        ):
            print(
                f"{i:>3} "
                f"| turn="
                f"{row['turn']:+8.3f} "
                f"| move="
                f"{row['move']:8.3f} "
                f"| MN9="
                f"{row['mn9']:>2} "
                f"| total="
                f"{row['total_spikes']}"
            )

    turn_mean, turn_std = (
        mean_std(
            [
                r["turn"]
                for r in baseline
            ]
        )
    )

    move_mean, move_std = (
        mean_std(
            [
                r["move"]
                for r in baseline
            ]
        )
    )

    total_mean, total_std = (
        mean_std(
            [
                r["total_spikes"]
                for r in baseline
            ]
        )
    )

    turn_tol = tolerance(
        turn_mean,
        turn_std,
    )

    move_tol = tolerance(
        move_mean,
        move_std,
    )

    total_tol = tolerance(
        total_mean,
        total_std,
        relative_floor=0.10,
    )

    print()
    print(
        "PRE-ATTACK OBSERVABLE SIGNATURE"
    )
    print(
        "--------------------------------"
    )

    print(
        f"TURN  mean="
        f"{turn_mean:+.3f} "
        f"std={turn_std:.3f} "
        f"tol=±{turn_tol:.3f}"
    )

    print(
        f"MOVE  mean="
        f"{move_mean:.3f} "
        f"std={move_std:.3f} "
        f"tol=±{move_tol:.3f}"
    )

    print(
        f"TOTAL mean="
        f"{total_mean:.1f} "
        f"std={total_std:.1f} "
        f"tol=±{total_tol:.1f}"
    )

    print(
        "MN9 baseline expected: 0"
    )

    # Clear only engineering filters.
    # MaleCNS is untouched.
    fly.reset_navigation_filters()

    print()
    print(
        "Running one complete +45° combat trial..."
    )

    telemetry = []

    result = run_trial(
        fly,
        trial_number=1,
        target_bearing=+45.0,
        telemetry=telemetry,
    )

    print()
    print(
        "Combat result:"
    )

    print(
        f"  success="
        f"{result['success']}"
    )

    print(
        f"  target attacks="
        f"{result['target_attack_events']}"
    )

    print(
        f"  accidental="
        f"{result['accidental_attacks']}"
    )

    if not result["success"]:

        print()
        print(
            "The prerequisite combat trial failed."
        )

        print(
            "Stop here; recovery timing would "
            "not be interpretable."
        )

        return

    print()
    print(
        "AttackDecoder is already re-armed here."
    )

    print(
        "Now testing whether the rest of the "
        "observable sensorimotor state has recovered."
    )

    print()

    window = deque(
        maxlen=ROLLING_WINDOW
    )

    stable_windows = 0

    first_matching_step = None
    stable_recovery_step = None

    for recovery_step in range(
        1,
        MAX_POST_ATTACK_STEPS + 1,
    ):

        row = sample(
            fly
        )

        window.append(
            row
        )

        if len(window) < ROLLING_WINDOW:
            continue

        turn_avg = float(
            np.mean(
                [
                    r["turn"]
                    for r in window
                ]
            )
        )

        move_avg = float(
            np.mean(
                [
                    r["move"]
                    for r in window
                ]
            )
        )

        total_avg = float(
            np.mean(
                [
                    r["total_spikes"]
                    for r in window
                ]
            )
        )

        mn9_clean = all(
            r["mn9"] == 0
            for r in window
        )

        no_attack_events = all(
            not r["attack_event"]
            for r in window
        )

        turn_ok = (
            abs(
                turn_avg
                - turn_mean
            )
            <= turn_tol
        )

        move_ok = (
            abs(
                move_avg
                - move_mean
            )
            <= move_tol
        )

        total_ok = (
            abs(
                total_avg
                - total_mean
            )
            <= total_tol
        )

        matching = (
            mn9_clean
            and
            no_attack_events
            and
            turn_ok
            and
            move_ok
            and
            total_ok
        )

        if matching:

            if (
                first_matching_step
                is None
            ):
                first_matching_step = (
                    recovery_step
                )

            stable_windows += 1

        else:

            stable_windows = 0

        if (
            recovery_step <= 30
            or
            recovery_step % 50 == 0
            or
            matching
        ):

            print(
                f"{recovery_step:>4} "
                f"| turn="
                f"{turn_avg:+8.3f} "
                f"{'✓' if turn_ok else 'X'} "
                f"| move="
                f"{move_avg:8.3f} "
                f"{'✓' if move_ok else 'X'} "
                f"| total="
                f"{total_avg:10.1f} "
                f"{'✓' if total_ok else 'X'} "
                f"| MN9="
                f"{'✓' if mn9_clean else 'X'} "
                f"| stable="
                f"{stable_windows}/"
                f"{STABLE_WINDOWS_REQUIRED}"
            )

        if (
            stable_windows
            >= STABLE_WINDOWS_REQUIRED
        ):

            stable_recovery_step = (
                recovery_step
            )

            break

    print()
    print(
        "=================================="
    )

    print(
        "FINAL RECOVERY RESULT"
    )

    print(
        "=================================="
    )

    print()

    if first_matching_step is None:

        print(
            "No observable baseline match "
            f"within {MAX_POST_ATTACK_STEPS} steps."
        )

        print(
            f"Equivalent simulated time: "
            f"{MAX_POST_ATTACK_STEPS * STEP_MS:.1f} ms"
        )

        print()

        print(
            "OBSERVABLE RECOVERY: FAIL ❌"
        )

    else:

        print(
            f"First matching rolling window: "
            f"step {first_matching_step}"
        )

        print(
            f"First-match simulated time: "
            f"{first_matching_step * STEP_MS:.1f} ms"
        )

        if stable_recovery_step is not None:

            print(
                f"Stable recovery reached: "
                f"step {stable_recovery_step}"
            )

            print(
                f"Stable recovery simulated time: "
                f"{stable_recovery_step * STEP_MS:.1f} ms"
            )

            print()

            print(
                "OBSERVABLE RECOVERY: PASS ✅"
            )

        else:

            print(
                "The signature matched temporarily "
                "but did not remain stable."
            )

            print()

            print(
                "OBSERVABLE RECOVERY: UNSTABLE ❌"
            )

    print()
    print(
        "Important:"
    )

    print(
        "This gate observes MN9, DNp20, DNpe017 "
        "and total network spikes."
    )

    print(
        "It does NOT prove every hidden MaleCNS "
        "state variable has returned exactly "
        "to its original state."
    )


if __name__ == "__main__":
    main()
