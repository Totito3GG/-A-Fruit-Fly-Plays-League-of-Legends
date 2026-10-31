from experiments.combat_arena_2d import (
    CombatFlyQ,
    TRIAL_BEARINGS,
    run_trial,
)


def main():

    print()
    print("FlyQ 35A — Fresh-Brain Trial Diagnostic")
    print("========================================")
    print()

    print(
        "Diagnostic only:"
    )

    print(
        "Each combat trial receives a fresh MaleCNS."
    )

    print(
        "Navigation/attack parameters are unchanged."
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "Determine whether 35A V1 failures are caused"
    )

    print(
        "by cross-trial post-attack neural carryover."
    )

    print()

    results = []

    total_neural_steps = 0

    for trial_number, bearing in enumerate(
        TRIAL_BEARINGS,
        start=1,
    ):

        print()
        print(
            "########################################"
        )

        print(
            f"FRESH MALECNS — TRIAL {trial_number}"
        )

        print(
            "########################################"
        )

        fly = CombatFlyQ()

        baseline = fly.calibrate_black(
            steps=30
        )

        print(
            f"Black DNp20 baseline: "
            f"{baseline:+.3f} Hz"
        )

        neural_start = (
            fly.neural_steps
        )

        telemetry = []

        result = run_trial(
            fly,
            trial_number,
            bearing,
            telemetry,
        )

        trial_neural_steps = (
            fly.neural_steps
            - neural_start
        )

        total_neural_steps += (
            trial_neural_steps
        )

        result[
            "neural_steps"
        ] = trial_neural_steps

        results.append(
            result
        )

    print()
    print(
        "========================================"
    )

    print(
        "FINAL DIAGNOSTIC SUMMARY"
    )

    print(
        "========================================"
    )

    print()

    successes = 0
    accidental = 0
    target_attacks = 0

    for result in results:

        if result["success"]:
            successes += 1

        accidental += (
            result[
                "accidental_attacks"
            ]
        )

        target_attacks += (
            result[
                "target_attack_events"
            ]
        )

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
            f"| neural="
            f"{result['neural_steps']} "
            f"| "
            f"{'✅' if result['success'] else '❌'}"
        )

    print()

    print(
        f"Successes: "
        f"{successes}/"
        f"{len(TRIAL_BEARINGS)}"
    )

    print(
        f"Target attacks: "
        f"{target_attacks}"
    )

    print(
        f"Accidental attacks: "
        f"{accidental}"
    )

    print(
        f"MaleCNS instances: "
        f"{len(TRIAL_BEARINGS)}"
    )

    print(
        f"Total experiment neural steps: "
        f"{total_neural_steps}"
    )

    print()

    if (
        successes
        == len(TRIAL_BEARINGS)
        and
        target_attacks
        == len(TRIAL_BEARINGS)
        and
        accidental == 0
    ):

        print(
            "DIAGNOSTIC RESULT: 6/6 ✅"
        )

        print()

        print(
            "Strong evidence:"
        )

        print(
            "35A V1 failure comes from persistent"
        )

        print(
            "cross-trial neural state after attack,"
        )

        print(
            "not from the frozen navigation controller."
        )

    else:

        print(
            "DIAGNOSTIC RESULT: NOT 6/6 ❌"
        )

        print()

        print(
            "There is also an integration/navigation"
        )

        print(
            "problem independent of cross-trial state."
        )


if __name__ == "__main__":
    main()
