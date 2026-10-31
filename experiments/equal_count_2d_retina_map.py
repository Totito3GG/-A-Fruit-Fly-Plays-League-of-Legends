import csv
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from neural.runtime import FlyQBrainRuntime


UPSTREAM = Path.home() / "FlyQ" / "upstream-doomfly"
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

OUTPUT = Path(
    "telemetry/equal_count_2d_retina_map.csv"
)

X_BINS = 10

Y_EDGES = np.asarray([
    0.0,
    1.0 / 3.0,
    2.0 / 3.0,
    1.0,
])

#
# IMPORTANT:
# Every tested region gets exactly 25 receptors.
# Regions with fewer than 25 are skipped.
#
N_ACTIVE = 25

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


def select_evenly(candidates, uv):
    #
    # Deterministic spatial selection.
    #
    order = np.lexsort((
        uv[candidates, 0],
        uv[candidates, 1],
    ))

    ordered = candidates[order]

    positions = np.linspace(
        0,
        len(ordered) - 1,
        N_ACTIVE,
    ).astype(int)

    return ordered[positions]


def run_region(selected, uv):
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

    for _ in range(WARMUP_STEPS):
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
            baseline_turn[-MEASURE_STEPS:]
        )
    )

    baseline_move = float(
        np.mean(
            baseline_move[-MEASURE_STEPS:]
        )
    )

    turns = []
    moves = []

    for _ in range(STIMULUS_STEPS):
        result = runtime.process_light(
            stimulus
        )

        turns.append(
            result["turn_raw_hz"]
        )

        moves.append(
            result["movement_raw_hz"]
        )

    turn = float(
        np.mean(
            turns[-MEASURE_STEPS:]
        )
    )

    move = float(
        np.mean(
            moves[-MEASURE_STEPS:]
        )
    )

    return {
        "delta_turn": (
            turn - baseline_turn
        ),

        "delta_move": (
            move - baseline_move
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
    y = uv[:, 1]

    x_edges = np.linspace(
        0.0,
        1.0,
        X_BINS + 1,
    )

    print()
    print("FlyQ Equal-Count 2D Retinal Map v2")
    print("==================================")
    print()

    print(
        f"Exactly {N_ACTIVE} receptors "
        f"per tested region @ "
        f"{INTENSITY:.2f}"
    )

    print(
        f"Regions with < {N_ACTIVE} "
        f"receptors are skipped."
    )

    print()

    pools = {}

    print("REGION POOL SIZES")
    print("=================")
    print()

    for yi in range(
        len(Y_EDGES) - 1
    ):
        y0 = Y_EDGES[yi]
        y1 = Y_EDGES[yi + 1]

        row = []

        for xi in range(X_BINS):
            x0 = x_edges[xi]
            x1 = x_edges[xi + 1]

            if xi == X_BINS - 1:
                xmask = (
                    (x >= x0)
                    & (x <= x1)
                )
            else:
                xmask = (
                    (x >= x0)
                    & (x < x1)
                )

            if yi == len(Y_EDGES) - 2:
                ymask = (
                    (y >= y0)
                    & (y <= y1)
                )
            else:
                ymask = (
                    (y >= y0)
                    & (y < y1)
                )

            candidates = np.flatnonzero(
                xmask & ymask
            )

            pools[(yi, xi)] = candidates

            row.append(
                len(candidates)
            )

        print(
            f"Y {y0:.2f}-{y1:.2f}: "
            + " ".join(
                f"{value:>4}"
                for value in row
            )
        )

    results = {}

    print()
    print("FUNCTIONAL MAP")
    print("==============")

    for yi in range(
        len(Y_EDGES) - 1
    ):
        y0 = Y_EDGES[yi]
        y1 = Y_EDGES[yi + 1]

        print()
        print(
            f"===== Y {y0:.2f}-{y1:.2f} ====="
        )
        print()

        for xi in range(X_BINS):
            x0 = x_edges[xi]
            x1 = x_edges[xi + 1]

            candidates = pools[
                (yi, xi)
            ]

            if len(candidates) < N_ACTIVE:
                results[(yi, xi)] = None

                print(
                    f"X {x0:.1f}-{x1:.1f} | "
                    f"pool={len(candidates):>3} | "
                    f"SKIP"
                )

                continue

            selected = select_evenly(
                candidates,
                uv,
            )

            result = run_region(
                selected,
                uv,
            )

            selected_sides = sides[
                selected
            ]

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

            state = classify(
                result["delta_turn"]
            )

            result.update({
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "pool": len(candidates),
                "L": n_left,
                "R": n_right,
                "state": state,
            })

            results[(yi, xi)] = result

            print(
                f"X {x0:.1f}-{x1:.1f} | "
                f"pool={len(candidates):>3} | "
                f"L={n_left:>2} "
                f"R={n_right:>2} | "
                f"Δturn="
                f"{result['delta_turn']:+7.2f} | "
                f"{state:>6} | "
                f"Δmove="
                f"{result['delta_move']:+7.2f}"
            )

    print()
    print("STEERING MAP")
    print("============")
    print()

    print(
        "       "
        + " ".join(
            f"{i/10:.1f}"
            for i in range(X_BINS)
        )
    )

    for yi in range(
        len(Y_EDGES) - 1
    ):
        symbols = []

        for xi in range(X_BINS):
            result = results[
                (yi, xi)
            ]

            if result is None:
                symbol = "X"

            else:
                symbol = {
                    "LEFT": "L",
                    "CENTER": ".",
                    "RIGHT": "R",
                }[
                    result["state"]
                ]

            symbols.append(
                symbol
            )

        print(
            f"Y{yi}:    "
            + "   ".join(symbols)
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid = [
        r
        for r in results.values()
        if r is not None
    ]

    with OUTPUT.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=valid[0].keys(),
        )

        writer.writeheader()
        writer.writerows(valid)

    print()
    print("Legend:")
    print("L = LEFT")
    print(". = CENTER")
    print("R = RIGHT")
    print("X = insufficient receptors")

    print()
    print(
        f"Saved → {OUTPUT}"
    )


if __name__ == "__main__":
    main()
