from pathlib import Path

import numpy as np
import pyarrow.feather as feather


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


def stats(name, values):
    values = np.asarray(
        values,
        dtype=float,
    )

    print()
    print(name)
    print("-" * len(name))

    print(f"N      : {len(values)}")

    if not len(values):
        return

    print(f"min    : {values.min():.6f}")
    print(f"max    : {values.max():.6f}")
    print(f"mean   : {values.mean():.6f}")
    print(f"median : {np.median(values):.6f}")

    for q in [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
    ]:
        print(
            f"q{q:.2f}  : "
            f"{np.quantile(values, q):.6f}"
        )


def main():
    graph = np.load(
        GRAPH,
        allow_pickle=False,
    )

    retina = graph["retina"]
    uv = graph["uv"]

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

    print()
    print("FlyQ Retina Anatomical-Side Audit")
    print("=================================")
    print()

    print(
        f"Mapped receptors: {len(retina)}"
    )

    unique, counts = np.unique(
        sides.astype(str),
        return_counts=True,
    )

    print()
    print("rootSide counts")
    print("---------------")

    for side, count in zip(
        unique,
        counts,
    ):
        print(
            f"{side:>8} : {count}"
        )

    for side in ["L", "R"]:
        mask = sides == side

        stats(
            f"{side} eye projected X",
            x[mask],
        )

        stats(
            f"{side} eye projected Y",
            y[mask],
        )

        left_count = int(
            np.sum(
                x[mask] < 0.4
            )
        )

        overlap_count = int(
            np.sum(
                (x[mask] >= 0.4)
                & (x[mask] <= 0.6)
            )
        )

        right_count = int(
            np.sum(
                x[mask] > 0.6
            )
        )

        print()
        print(
            f"{side} screen occupancy"
        )
        print(
            "-" * (
                len(side)
                + len(" screen occupancy")
            )
        )

        print(
            f"x < .4  : {left_count}"
        )

        print(
            f".4-.6   : {overlap_count}"
        )

        print(
            f"x > .6  : {right_count}"
        )

    lmask = sides == "L"
    rmask = sides == "R"

    l_internal = (
        x[lmask] / 0.60
    )

    r_internal = (
        1.0
        - (
            (x[rmask] - 0.40)
            / 0.60
        )
    )

    stats(
        "L inferred internal horizontal coordinate",
        l_internal,
    )

    stats(
        "R inferred internal horizontal coordinate",
        r_internal,
    )

    bins = np.linspace(
        0.0,
        1.0,
        11,
    )

    lh, _ = np.histogram(
        l_internal,
        bins=bins,
    )

    rh, _ = np.histogram(
        r_internal,
        bins=bins,
    )

    print()
    print("Internal-coordinate comparison")
    print("------------------------------")

    for i in range(
        len(bins) - 1
    ):
        print(
            f"{bins[i]:.1f}-"
            f"{bins[i+1]:.1f} | "
            f"L={lh[i]:>4} | "
            f"R={rh[i]:>4} | "
            f"diff={int(rh[i]-lh[i]):+5d}"
        )

    total_l = int(
        np.sum(lmask)
    )

    total_r = int(
        np.sum(rmask)
    )

    print()
    print("SUMMARY")
    print("=======")

    print(
        f"Mapped anatomical L : "
        f"{total_l}"
    )

    print(
        f"Mapped anatomical R : "
        f"{total_r}"
    )

    if total_l:
        print(
            f"R/L count ratio      : "
            f"{total_r / total_l:.3f}"
        )


if __name__ == "__main__":
    main()
