import numpy as np

from neural.runtime import FlyQBrainRuntime


BINS = 20


def main():
    runtime = FlyQBrainRuntime()

    uv = np.asarray(
        runtime.brain.uv,
        dtype=float,
    )

    x = uv[:, 0]
    y = uv[:, 1]

    print()
    print("FlyQ Retina UV Audit")
    print("====================")
    print()

    print(f"Receptors : {len(uv)}")
    print()

    print("X statistics")
    print("------------")
    print(f"min    : {x.min():.6f}")
    print(f"max    : {x.max():.6f}")
    print(f"mean   : {x.mean():.6f}")
    print(f"median : {np.median(x):.6f}")

    print()
    print("Y statistics")
    print("------------")
    print(f"min    : {y.min():.6f}")
    print(f"max    : {y.max():.6f}")
    print(f"mean   : {y.mean():.6f}")
    print(f"median : {np.median(y):.6f}")

    left = np.sum(x < 0.5)
    right = np.sum(x > 0.5)
    middle = np.sum(x == 0.5)

    print()
    print("Horizontal split")
    print("----------------")
    print(f"x < .5  : {left}")
    print(f"x = .5  : {middle}")
    print(f"x > .5  : {right}")

    edges = np.linspace(
        0.0,
        1.0,
        BINS + 1,
    )

    counts, _ = np.histogram(
        x,
        bins=edges,
    )

    print()
    print("Horizontal receptor density")
    print("---------------------------")

    max_count = max(counts)

    for i, count in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]

        bar_len = int(
            40 * count / max_count
        )

        print(
            f"{lo:.2f}-{hi:.2f} | "
            f"{count:>4} | "
            f"{'#' * bar_len}"
        )

    print()
    print("Mirror symmetry test")
    print("--------------------")

    mirror_errors = []

    for i in range(BINS // 2):
        a = counts[i]
        b = counts[-1 - i]

        total = a + b

        if total:
            asymmetry = (
                abs(a - b) / total
            )
        else:
            asymmetry = 0.0

        mirror_errors.append(
            asymmetry
        )

        print(
            f"bin {i:>2} vs {BINS-1-i:>2} | "
            f"{a:>4} vs {b:>4} | "
            f"asym={asymmetry:.3f}"
        )

    print()
    print(
        "Mean mirror asymmetry : "
        f"{np.mean(mirror_errors):.4f}"
    )

    print()
    print("UV quantiles")
    print("------------")

    for q in [
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]:
        print(
            f"x q{q:>4.2f} : "
            f"{np.quantile(x, q):.6f}"
        )


if __name__ == "__main__":
    main()
