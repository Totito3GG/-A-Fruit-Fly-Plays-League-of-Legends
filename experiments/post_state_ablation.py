import numpy as np

import experiments.combat_arena_2d as combat


BASELINE_STEPS = 50

# Long enough to detect the transition we previously
# saw around ~200-250 recovery steps.
POST_STEPS = 1000

# Comparable long-running no-stimulus control.
BLACK_EXPOSURE_STEPS = 400

SUGAR_STEPS = 2

TARGET_BEARING = +45.0


def sample(fly, sugar=False):
    obs = fly.observe(
        combat.black_frame(),
        sugar=sugar,
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
        "total": int(
            fly.brain.counts.sum()
        ),
    }


def collect_baseline(fly):
    values = []

    fly.calibrate_black(
        steps=30
    )

    for _ in range(
        BASELINE_STEPS
    ):
        values.append(
            sample(fly)
        )

    return values


def mean(values, key):
    return float(
        np.mean(
            [
                v[key]
                for v in values
            ]
        )
    )


def summarize_window(
    name,
    values,
):
    print(
        f"{name:<14}"
        f" | total="
        f"{mean(values, 'total'):9.1f}"
        f" | turn="
        f"{mean(values, 'turn'):+8.3f}"
        f" | move="
        f"{mean(values, 'move'):8.3f}"
        f" | MN9-active="
        f"{sum(v['mn9'] > 0 for v in values):>3}"
        f" | attacks="
        f"{sum(v['attack_event'] for v in values):>3}"
    )


def collect_post(
    fly,
):
    values = []

    checkpoints = {
        50,
        100,
        200,
        250,
        400,
        600,
        800,
        1000,
    }

    rolling = []

    for step in range(
        1,
        POST_STEPS + 1,
    ):
        row = sample(
            fly,
            sugar=False,
        )

        values.append(
            row
        )

        rolling.append(
            row
        )

        if len(rolling) > 50:
            rolling.pop(0)

        if (
            step in checkpoints
        ):
            print(
                f"post {step:>4}"
                f" | total="
                f"{mean(rolling, 'total'):9.1f}"
                f" | turn="
                f"{mean(rolling, 'turn'):+8.3f}"
                f" | move="
                f"{mean(rolling, 'move'):8.3f}"
                f" | MN9active="
                f"{sum(v['mn9'] > 0 for v in rolling):>2}/"
                f"{len(rolling)}"
            )

    return values


def condition_black_only():

    print()
    print(
        "========================================"
    )
    print(
        "A — BLACK-ONLY CONTROL"
    )
    print(
        "========================================"
    )

    fly = combat.CombatFlyQ()

    baseline = collect_baseline(
        fly
    )

    summarize_window(
        "baseline",
        baseline,
    )

    print(
        f"\nApplying "
        f"{BLACK_EXPOSURE_STEPS} "
        f"additional black steps..."
    )

    for _ in range(
        BLACK_EXPOSURE_STEPS
    ):
        sample(
            fly,
            sugar=False,
        )

    print(
        "\nPost-exposure black monitoring:"
    )

    post = collect_post(
        fly
    )

    return {
        "name": "BLACK_ONLY",
        "baseline": baseline,
        "post": post,
    }


def condition_sugar_only():

    print()
    print(
        "========================================"
    )
    print(
        "B — SUGAR-ONLY"
    )
    print(
        "========================================"
    )

    fly = combat.CombatFlyQ()

    baseline = collect_baseline(
        fly
    )

    summarize_window(
        "baseline",
        baseline,
    )

    print()
    print(
        f"Applying engineered sugar pulse: "
        f"{SUGAR_STEPS} steps "
        f"({SUGAR_STEPS * combat.STEP_MS:.1f} ms)"
    )

    for step in range(
        1,
        SUGAR_STEPS + 1,
    ):
        row = sample(
            fly,
            sugar=True,
        )

        print(
            f"sugar {step}"
            f" | total="
            f"{row['total']}"
            f" | MN9="
            f"{row['mn9']}"
            f" | event="
            f"{row['attack_event']}"
        )

    print(
        "\nPost-sugar black monitoring:"
    )

    post = collect_post(
        fly
    )

    return {
        "name": "SUGAR_ONLY",
        "baseline": baseline,
        "post": post,
    }


def condition_navigation_only():

    print()
    print(
        "========================================"
    )
    print(
        "C — NAVIGATION-ONLY"
    )
    print(
        "========================================"
    )

    fly = combat.CombatFlyQ()

    baseline = collect_baseline(
        fly
    )

    summarize_window(
        "baseline",
        baseline,
    )

    print()
    print(
        "Running frozen +45° navigation."
    )
    print(
        "When target range is reached:"
    )
    print(
        "  NO sugar"
    )
    print(
        "  NO MN9 attack stimulation"
    )
    print()

    original_attack = (
        combat.perform_attack
    )

    def no_attack(_fly):
        print(
            "      [ABLATION] "
            "attack stimulus suppressed"
        )

        return {
            "events": 0,
            "pulse_steps": 0,
            "wait_steps": 0,
            "recovery_steps": 0,
            "rearmed": True,
        }

    combat.perform_attack = (
        no_attack
    )

    try:

        result = combat.run_trial(
            fly,
            trial_number=1,
            target_bearing=TARGET_BEARING,
            telemetry=[],
        )

    finally:

        combat.perform_attack = (
            original_attack
        )

    print()
    print(
        "Navigation-only endpoint:"
    )

    print(
        f"  min distance="
        f"{result['min_distance']:.3f}"
    )

    print(
        f"  attack events="
        f"{result['target_attack_events']}"
    )

    print(
        f"  accidental attacks="
        f"{result['accidental_attacks']}"
    )

    print(
        "\nPost-navigation black monitoring:"
    )

    post = collect_post(
        fly
    )

    return {
        "name": "NAVIGATION_ONLY",
        "baseline": baseline,
        "post": post,
    }


def final_summary(
    results,
):

    print()
    print(
        "========================================"
    )
    print(
        "FINAL ABLATION SUMMARY"
    )
    print(
        "========================================"
    )
    print()

    for result in results:

        baseline = result[
            "baseline"
        ]

        post = result[
            "post"
        ]

        early = post[
            :50
        ]

        middle = post[
            200:250
        ]

        late = post[
            -50:
        ]

        baseline_total = mean(
            baseline,
            "total",
        )

        early_total = mean(
            early,
            "total",
        )

        middle_total = mean(
            middle,
            "total",
        )

        late_total = mean(
            late,
            "total",
        )

        late_ratio = (
            late_total
            / baseline_total
        )

        late_mn9 = sum(
            v["mn9"] > 0
            for v in late
        )

        print(
            f"{result['name']:<16}"
            f" | baseline="
            f"{baseline_total:9.1f}"
            f" | early="
            f"{early_total:9.1f}"
            f" | mid="
            f"{middle_total:9.1f}"
            f" | late="
            f"{late_total:9.1f}"
            f" | late/base="
            f"{late_ratio:5.2f}x"
            f" | late MN9="
            f"{late_mn9}/50"
        )

    print()
    print(
        "Interpretation:"
    )
    print()

    print(
        "BLACK_ONLY high"
        "  -> persistent model/runtime drift."
    )

    print(
        "SUGAR_ONLY high, BLACK stable"
        "  -> sugar pathway sufficient."
    )

    print(
        "NAVIGATION_ONLY high, BLACK stable"
        "  -> navigation stimulation sufficient."
    )

    print(
        "SUGAR + NAV each stable individually"
        "  -> interaction/synergy likely."
    )

    print()
    print(
        "Do not call any mechanism biological"
    )
    print(
        "until the causal condition is isolated."
    )


def main():

    print()
    print(
        "FlyQ Post-State Causal Ablation"
    )
    print(
        "================================"
    )

    print()
    print(
        "Fresh MaleCNS per CONDITION."
    )
    print(
        "Persistent MaleCNS within each condition."
    )
    print(
        "No synaptic learning/plasticity enabled."
    )

    results = []

    results.append(
        condition_black_only()
    )

    results.append(
        condition_sugar_only()
    )

    results.append(
        condition_navigation_only()
    )

    final_summary(
        results
    )


if __name__ == "__main__":
    main()
