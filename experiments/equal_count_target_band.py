from pathlib import Path

import numpy as np

from neural.runtime import FlyQBrainRuntime


UPSTREAM = (
    Path.home()
    / "FlyQ"
    / "upstream-doomfly"
)

GRAPH = (
    UPSTREAM
    / "outputs"
    / "doom"
    / "malecns_v1"
    / "graph.npz"
)


BINS = 10

INTENSITY = 0.5

WARMUP = 20
STIMULUS = 20
MEASURE = 7

#
# Same approximate vertical field used by
# the previous 160px-high target in a
# 240px image.
#
Y_MIN = 40 / 239
Y_MAX = 199 / 239


def classify(value):
    if value < -3:
        return "LEFT"

    if value > 3:
        return "RIGHT"

    return "CENTER"


def run(selected, uv):
    runtime = FlyQBrainRuntime()

    black = np.zeros(
        len(uv),
        dtype=np.float32,
    )

    light = np.zeros(
        len(uv),
        dtype=np.float32,
    )

    light[selected] = INTENSITY

    baseline = []

    for _ in range(WARMUP):
        result = runtime.process_light(
            black
        )

        baseline.append(
            result["turn_raw_hz"]
        )

    baseline = float(
        np.mean(
            baseline[-MEASURE:]
        )
    )

    turns = []

    for _ in range(STIMULUS):
        result = runtime.process_light(
            light
        )

        turns.append(
            result["turn_raw_hz"]
        )

    turn = float(
        np.mean(
            turns[-MEASURE:]
        )
    )

    return turn - baseline


def main():
    graph = np.load(
        GRAPH,
        allow_pickle=False,
    )

    uv = np.asarray(
        graph["uv"],
        dtype=float,
    )

    graph.close()

    x = uv[:, 0]
    y = uv[:, 1]

    edges = np.linspace(
        0,
        1,
        BINS + 1,
    )

    pools = []

    for i in range(BINS):
        lo = edges[i]
        hi = edges[i + 1]

        if i == BINS - 1:
            mask = (
                (x >= lo)
                & (x <= hi)
            )
        else:
            mask = (
                (x >= lo)
                & (x < hi)
            )

        mask &= (
            (y >= Y_MIN)
            & (y <= Y_MAX)
        )

        pools.append(
            np.flatnonzero(mask)
        )

    #
    # Equal count in every region.
    #
    n_active = min(
        80,
        min(len(p) for p in pools),
    )

    print()
    print(
        "FlyQ Equal-Count Target-Band Map"
    )
    print(
        "================================"
    )
    print()

    print(
        f"Vertical range : "
        f"{Y_MIN:.3f} – {Y_MAX:.3f}"
    )

    print(
        f"Active/bin     : {n_active}"
    )

    print()

    for i, candidates in enumerate(pools):

        lo = edges[i]
        hi = edges[i + 1]

        #
        # Spread selected receptors through
        # the available vertical range.
        #
        ordered = candidates[
            np.argsort(
                y[candidates]
            )
        ]

        positions = np.linspace(
            0,
            len(ordered) - 1,
            n_active,
        ).astype(int)

        selected = ordered[
            positions
        ]

        delta = run(
            selected,
            uv,
        )

        print(
            f"{lo:.1f}-{hi:.1f} | "
            f"pool={len(candidates):>3} | "
            f"N={n_active:>3} | "
            f"Δturn={delta:+7.2f} | "
            f"{classify(delta):>6}"
        )


if __name__ == "__main__":
    main()
