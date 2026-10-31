from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from neural.runtime import FlyQBrainRuntime


UPSTREAM = (
    Path.home()
    / "FlyQ"
    / "upstream-doomfly"
)

DATASET = "malecns_v1"

GRAPH = (
    UPSTREAM
    / "outputs"
    / "doom"
    / DATASET
    / "graph.npz"
)

DATA = (
    UPSTREAM
    / "connectome_data"
    / DATASET
)


BINS = 10

N_ACTIVE = 100
INTENSITY = 0.5

WARMUP_STEPS = 20
STIMULUS_STEPS = 20
MEASURE_STEPS = 7


def classify(delta_turn):
    if delta_turn < -3.0:
        return "LEFT"

    if delta_turn > 3.0:
        return "RIGHT"

    return "CENTER"


def choose_evenly_in_y(
    indices,
    uv,
    n,
):
    #
    # Deterministic selection distributed
    # vertically through the bin instead of
    # taking arbitrary graph order.
    #
    ordered = indices[
        np.argsort(
            uv[indices, 1]
        )
    ]

    positions = np.linspace(
        0,
        len(ordered) - 1,
        n,
    ).astype(int)

    return ordered[positions]


def run_bin(
    selected,
    uv,
):
    runtime = FlyQBrainRuntime()

    if not np.allclose(
        runtime.brain.uv,
        uv,
    ):
        raise RuntimeError(
            "Runtime UV order differs "
            "from graph UV order."
        )

    black = np.zeros(
        len(uv),
        dtype=np.float32,
    )

    stimulus = np.zeros(
        len(uv),
        dtype=np.float32,
    )

    stimulus[selected] = INTENSITY

    baseline_turn = []
    baseline_move = []

    for _ in range(
        WARMUP_STEPS
    ):
        result = runtime.process_light(
            black
        )

        baseline_turn.append(
            result["turn_raw_hz"]
        )

        baseline_move.append(
            result["movement_raw_hz"]
        )

    baseline_turn = float(
        np.mean(
            baseline_turn[
                -MEASURE_STEPS:
            ]
        )
    )

    baseline_move = float(
        np.mean(
            baseline_move[
                -MEASURE_STEPS:
            ]
        )
    )

    turns = []
    movements = []

    for _ in range(
        STIMULUS_STEPS
    ):
        result = runtime.process_light(
            stimulus
        )

        turns.append(
            result["turn_raw_hz"]
        )

        movements.append(
            result["movement_raw_hz"]
        )

    turn = float(
        np.mean(
            turns[
                -MEASURE_STEPS:
            ]
        )
    )

    movement = float(
        np.mean(
            movements[
                -MEASURE_STEPS:
            ]
        )
    )

    return {
        "turn": turn,

        "delta_turn": (
            turn - baseline_turn
        ),

        "movement": movement,

        "delta_move": (
            movement - baseline_move
        ),
    }


def main():
    graph = np.load(
        GRAPH,
        allow_pickle=False,
    )

    retina = graph["retina"]

    uv = np.asarray(
        graph["uv"],
        dtype=np.float32,
    )

    graph.close()

    nodes = feather.read_table(
        DATA
        / "normalized"
        / "neurons.feather"
    ).to_pandas()

    annotations = feather.read_table(
        DATA
        / "annotations.feather"
    ).to_pandas()

    annotations = (
        annotations
        .set_index("bodyId")
        .loc[nodes.source_id]
    )

    sides = (
        annotations.rootSide
        .to_numpy()[retina]
    )

    x = uv[:, 0]

    edges = np.linspace(
        0.0,
        1.0,
        BINS + 1,
    )

    print()
    print("FlyQ Equal-Count Retinal Map")
    print("============================")
    print()

    print(
        f"Every condition activates exactly "
        f"{N_ACTIVE} receptors @ "
        f"{INTENSITY:.2f}"
    )

    print()

    results = []

    for bin_id in range(BINS):

        lo = edges[bin_id]
        hi = edges[bin_id + 1]

        if bin_id == BINS - 1:
            candidates = np.flatnonzero(
                (x >= lo)
                & (x <= hi)
            )
        else:
            candidates = np.flatnonzero(
                (x >= lo)
                & (x < hi)
            )

        if len(candidates) < N_ACTIVE:
            raise RuntimeError(
                f"Bin {lo:.1f}-{hi:.1f} "
                f"contains only "
                f"{len(candidates)} receptors."
            )

        selected = choose_evenly_in_y(
            candidates,
            uv,
            N_ACTIVE,
        )

        selected_sides = sides[selected]

        n_left = int(
            np.sum(
                selected_sides == "L"
            )
        )

        n_right = int(
            np.sum(
                selected_sides == "R"
            )
        )

        other = (
            N_ACTIVE
            - n_left
            - n_right
        )

        result = run_bin(
            selected,
            uv,
        )

        result.update({
            "lo": lo,
            "hi": hi,

            "candidates": (
                len(candidates)
            ),

            "mean_x": float(
                uv[selected, 0].mean()
            ),

            "mean_y": float(
                uv[selected, 1].mean()
            ),

            "L": n_left,
            "R": n_right,
            "other": other,

            "state": classify(
                result["delta_turn"]
            ),
        })

        results.append(
            result
        )

    print("RESULTS")
    print("=======")
    print()

    for r in results:

        print(
            f"{r['lo']:.1f}-"
            f"{r['hi']:.1f} | "
            f"pool={r['candidates']:>3} | "
            f"L={r['L']:>3} "
            f"R={r['R']:>3} | "
            f"x̄={r['mean_x']:.3f} | "
            f"Δturn={r['delta_turn']:+7.2f} | "
            f"{r['state']:>6} | "
            f"Δmove={r['delta_move']:+7.2f}"
        )

    print()
    print("LEFT-ONLY REGION")
    print("================")
    print()

    for r in results:
        if (
            r["L"] == N_ACTIVE
            and r["R"] == 0
        ):
            print(
                f"{r['lo']:.1f}-"
                f"{r['hi']:.1f} → "
                f"Δturn="
                f"{r['delta_turn']:+7.2f} "
                f"{r['state']}"
            )

    print()
    print("RIGHT-ONLY REGION")
    print("=================")
    print()

    for r in results:
        if (
            r["R"] == N_ACTIVE
            and r["L"] == 0
        ):
            print(
                f"{r['lo']:.1f}-"
                f"{r['hi']:.1f} → "
                f"Δturn="
                f"{r['delta_turn']:+7.2f} "
                f"{r['state']}"
            )


if __name__ == "__main__":
    main()
