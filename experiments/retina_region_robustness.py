from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime


UPSTREAM = Path.home() / "FlyQ" / "upstream-doomfly"

GRAPH = (
    UPSTREAM
    / "outputs"
    / "doom"
    / "malecns_v1"
    / "graph.npz"
)


N_ACTIVE = 25
INTENSITY = 0.5

REPEATS = 5

WARMUP_STEPS = 20
STIMULUS_STEPS = 20
MEASURE_STEPS = 7

SEED = 12345


#
# (name, x0, x1, y0, y1)
#
REGIONS = [
    (
        "top_far_left",
        0.0, 0.1,
        0.0, 1/3,
    ),

    (
        "top_left_mid",
        0.2, 0.3,
        0.0, 1/3,
    ),

    #
    # The important anomalous region.
    #
    (
        "middle_left_anomaly",
        0.1, 0.2,
        1/3, 2/3,
    ),

    (
        "middle_left_inner",
        0.3, 0.4,
        1/3, 2/3,
    ),

    (
        "top_far_right",
        0.8, 0.9,
        0.0, 1/3,
    ),

    (
        "middle_far_right",
        0.8, 0.9,
        1/3, 2/3,
    ),

    (
        "bottom_far_right",
        0.8, 0.9,
        2/3, 1.0,
    ),
]


def classify(value):
    if value < -3.0:
        return "LEFT"

    if value > 3.0:
        return "RIGHT"

    return "CENTER"


def run(selected, uv):
    runtime = FlyQBrainRuntime()

    black = np.zeros(
        len(uv),
        dtype=np.float32,
    )

    stimulus = np.zeros(
        len(uv),
        dtype=np.float32,
    )

    stimulus[selected] = INTENSITY

    baseline = []

    for _ in range(
        WARMUP_STEPS
    ):
        result = runtime.process_light(
            black
        )

        baseline.append(
            result["turn_raw_hz"]
        )

    baseline = float(
        np.mean(
            baseline[-MEASURE_STEPS:]
        )
    )

    turns = []

    for _ in range(
        STIMULUS_STEPS
    ):
        result = runtime.process_light(
            stimulus
        )

        turns.append(
            result["turn_raw_hz"]
        )

    turn = float(
        np.mean(
            turns[-MEASURE_STEPS:]
        )
    )

    return turn - baseline


def region_candidates(
    uv,
    x0,
    x1,
    y0,
    y1,
):
    x = uv[:, 0]
    y = uv[:, 1]

    xmask = (
        (x >= x0)
        & (x < x1)
    )

    if x1 == 1.0:
        xmask = (
            (x >= x0)
            & (x <= x1)
        )

    ymask = (
        (y >= y0)
        & (y < y1)
    )

    if y1 == 1.0:
        ymask = (
            (y >= y0)
            & (y <= y1)
        )

    return np.flatnonzero(
        xmask & ymask
    )


def main():
    graph = np.load(
        GRAPH,
        allow_pickle=False,
    )

    uv = np.asarray(
        graph["uv"],
        dtype=np.float32,
    )

    graph.close()

    rng = np.random.default_rng(
        SEED
    )

    print()
    print(
        "FlyQ Retinal Region Robustness"
    )
    print(
        "=============================="
    )

    print()
    print(
        f"{REPEATS} different receptor subsets"
        f" per region"
    )

    print(
        f"{N_ACTIVE} receptors/subset "
        f"@ {INTENSITY:.2f}"
    )

    for (
        name,
        x0,
        x1,
        y0,
        y1,
    ) in REGIONS:

        candidates = region_candidates(
            uv,
            x0,
            x1,
            y0,
            y1,
        )

        print()
        print(
            f"===== {name} ====="
        )

        print(
            f"x={x0:.2f}-{x1:.2f}, "
            f"y={y0:.2f}-{y1:.2f}, "
            f"pool={len(candidates)}"
        )

        if len(candidates) < N_ACTIVE:
            print(
                "SKIP: insufficient receptors"
            )
            continue

        values = []

        states = []

        for repeat in range(
            1,
            REPEATS + 1,
        ):

            selected = rng.choice(
                candidates,
                size=N_ACTIVE,
                replace=False,
            )

            delta = run(
                selected,
                uv,
            )

            state = classify(
                delta
            )

            values.append(
                delta
            )

            states.append(
                state
            )

            print(
                f"{repeat:>2} | "
                f"Δturn={delta:+7.2f} | "
                f"{state}"
            )

        values = np.asarray(
            values,
            dtype=float,
        )

        counts = {
            state: states.count(state)
            for state in [
                "LEFT",
                "CENTER",
                "RIGHT",
            ]
        }

        print(
            f"mean={values.mean():+.2f} "
            f"std={values.std():.2f} | "
            f"L/C/R="
            f"{counts['LEFT']}/"
            f"{counts['CENTER']}/"
            f"{counts['RIGHT']}"
        )


if __name__ == "__main__":
    main()
