import numpy as np

from experiments.combat_arena_2d import (
    CombatFlyQ,
    black_frame,
    STEP_MS,
)


BASELINE_STEPS = 50
POST_STEPS = 1000

DOSES = [
    0,
    1,
    2,
]

CHECKPOINTS = {
    10,
    20,
    50,
    100,
    200,
    250,
    400,
    600,
    800,
    1000,
}


def sample(
    fly,
    sugar=False,
):
    obs = fly.observe(
        black_frame(),
        sugar=sugar,
    )

    return {
        "total": int(
            fly.brain.counts.sum()
        ),

        "turn": float(
            obs["turn_raw_hz"]
        ),

        "move": float(
            obs["movement_raw_hz"]
        ),

        "mn9": int(
            obs["mn9_spikes"]
        ),

        "attack": bool(
            obs["attack_event"]
        ),
    }


def mean(
    values,
    key,
):
    return float(
        np.mean(
            [
                v[key]
                for v in values
            ]
        )
    )


def run_condition(
    dose,
):

    print()
    print(
        "========================================"
    )

    print(
        f"SUGAR DOSE = {dose} STEP(S)"
    )

    print(
        f"duration = "
        f"{dose * STEP_MS:.1f} ms"
    )

    print(
        "========================================"
    )

    fly = CombatFlyQ()

    fly.calibrate_black(
        steps=30
    )

    baseline = []

    for _ in range(
        BASELINE_STEPS
    ):
        baseline.append(
            sample(
                fly,
                sugar=False,
            )
        )

    baseline_total = mean(
        baseline,
        "total",
    )

    print(
        f"Baseline total: "
        f"{baseline_total:.1f}"
    )

    print(
        f"Baseline MN9 active: "
        f"{sum(x['mn9'] > 0 for x in baseline)}"
    )

    pulse_events = 0

    if dose > 0:

        print()
        print(
            "Applying sugar..."
        )

        for step in range(
            1,
            dose + 1,
        ):

            row = sample(
                fly,
                sugar=True,
            )

            pulse_events += int(
                row["attack"]
            )

            print(
                f"sugar {step}"
                f" | total="
                f"{row['total']}"
                f" | MN9="
                f"{row['mn9']}"
                f" | attack="
                f"{row['attack']}"
            )

    else:

        print()
        print(
            "No sugar applied."
        )

    post = []

    delayed_events = 0

    first_attack_step = None

    first_2x_step = None
    first_4x_step = None

    rolling = []

    print()
    print(
        "Post-stimulus monitoring..."
    )

    for step in range(
        1,
        POST_STEPS + 1,
    ):

        row = sample(
            fly,
            sugar=False,
        )

        post.append(
            row
        )

        rolling.append(
            row
        )

        if len(rolling) > 50:
            rolling.pop(0)

        if row["attack"]:

            delayed_events += 1

            if (
                first_attack_step
                is None
            ):
                first_attack_step = (
                    step
                )

        ratio_now = (
            row["total"]
            / baseline_total
        )

        if (
            first_2x_step
            is None
            and
            ratio_now >= 2.0
        ):
            first_2x_step = step

        if (
            first_4x_step
            is None
            and
            ratio_now >= 4.0
        ):
            first_4x_step = step

        if step in CHECKPOINTS:

            rolling_total = mean(
                rolling,
                "total",
            )

            rolling_ratio = (
                rolling_total
                / baseline_total
            )

            mn9_active = sum(
                r["mn9"] > 0
                for r in rolling
            )

            print(
                f"post {step:>4}"
                f" | total="
                f"{rolling_total:9.1f}"
                f" | ratio="
                f"{rolling_ratio:5.2f}x"
                f" | MN9="
                f"{mn9_active:>2}/"
                f"{len(rolling)}"
                f" | attacks="
                f"{sum(r['attack'] for r in rolling)}"
            )

    early = mean(
        post[:50],
        "total",
    )

    late = mean(
        post[-50:],
        "total",
    )

    late_ratio = (
        late
        / baseline_total
    )

    total_events = (
        pulse_events
        + delayed_events
    )

    return {
        "dose":
            dose,

        "duration_ms":
            dose * STEP_MS,

        "baseline":
            baseline_total,

        "early":
            early,

        "late":
            late,

        "late_ratio":
            late_ratio,

        "pulse_events":
            pulse_events,

        "delayed_events":
            delayed_events,

        "total_events":
            total_events,

        "first_attack":
            first_attack_step,

        "first_2x":
            first_2x_step,

        "first_4x":
            first_4x_step,
    }


def main():

    print()
    print(
        "FlyQ 35A-V2c — Sugar Dose Response"
    )

    print(
        "=================================="
    )

    print()

    print(
        "Fresh MaleCNS per dose."
    )

    print(
        "Persistent MaleCNS within each dose."
    )

    print(
        "No learning/plasticity enabled."
    )

    results = []

    for dose in DOSES:

        results.append(
            run_condition(
                dose
            )
        )

    print()
    print(
        "=================================="
    )

    print(
        "FINAL DOSE-RESPONSE SUMMARY"
    )

    print(
        "=================================="
    )

    print()

    for r in results:

        print(
            f"{r['dose']} step(s)"
            f" | {r['duration_ms']:5.1f} ms"
            f" | baseline="
            f"{r['baseline']:8.1f}"
            f" | early="
            f"{r['early']:8.1f}"
            f" | late="
            f"{r['late']:8.1f}"
            f" | late/base="
            f"{r['late_ratio']:5.2f}x"
            f" | attacks="
            f"{r['total_events']}"
            f" | first attack="
            f"{r['first_attack']}"
            f" | first 2x="
            f"{r['first_2x']}"
            f" | first 4x="
            f"{r['first_4x']}"
        )

    print()
    print(
        "Decision guide:"
    )

    print()

    print(
        "1-step attack works AND late/base ~1x:"
    )

    print(
        "  -> use 1-step engineered trigger."
    )

    print()

    print(
        "1-step attack works BUT late/base >>1x:"
    )

    print(
        "  -> even minimal sugar destabilizes "
        "the persistent sensorimotor state."
    )

    print()

    print(
        "1-step stable BUT produces no attack:"
    )

    print(
        "  -> sugar cannot currently provide both "
        "repeatability and clean persistence."
    )

    print()

    print(
        "0-step also becomes high:"
    )

    print(
        "  -> contradicts the previous ablation "
        "and requires runtime investigation."
    )


if __name__ == "__main__":
    main()
