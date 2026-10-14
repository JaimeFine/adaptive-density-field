"""Regenerate the ``nlist`` failure-mode figure from the sweep outputs.

Sources of truth
----------------
``data/ctu_adf_poi_nlist_<n>.csv`` for n in {64, 256, 1024, 4096, 16384}. Each
file holds all 82,651 evaluation points with the field value (``ADF``) and the
per-trajectory selection flag (``POI``).

What the figure shows
---------------------
Panel (a): points selected. Flat at 20,769 across 64--4096, then 32,105 at
16,384 -- more points selected without any gain in detection.
Panel (b): trajectories whose field takes a single distinct value, so the
per-trajectory threshold has nothing to separate. Zero across 64--4096, 45 at
16,384.

Both quantities are quoted in Section 4.4 and are printed here so the
manuscript text can be checked against them. The figure supports the reading
the text already gives: at 16,384 the configuration stops retrieving a query's
actual neighbours, which is a retrieval failure rather than a demonstration of
sensitivity to the partition size.

Note that the swept values are spaced by a factor of four, so the curve
between 4,096 and 16,384 is not resolved; the figure should not be read as
locating the breakdown point more precisely than that.

Outputs
-------
``results/nlist_failure.png`` and ``results/nlist_failure.pdf``

Usage:
    python nlist_failure_plot.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"

SWEEP = (64, 256, 1024, 4096, 16384)
STABLE_SELECTED = 20769

COLOUR_A = "#12406b"
COLOUR_B = "#b2182b"


def summarise(nlist: int) -> dict:
    df = pd.read_csv(DATA / f"ctu_adf_poi_nlist_{nlist}.csv")
    distinct = df.groupby("flight_id")["ADF"].nunique()
    return dict(
        nlist=nlist,
        selected=int((df["POI"] == 1).sum()),
        collapsed=int((distinct == 1).sum()),
        n_flights=int(len(distinct)),
    )


def main() -> None:
    rows = [summarise(n) for n in SWEEP]
    df = pd.DataFrame(rows)

    print(f"{'nlist':>7}{'selected':>10}{'collapsed (1 distinct value)':>32}"
          f"{'of flights':>12}")
    for r in rows:
        print(f"{r['nlist']:>7,}{r['selected']:>10,}{r['collapsed']:>32,}"
              f"{r['n_flights']:>12,}")

    x = df["nlist"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.2, 3.9), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.18),
    )

    # (a) selected points ---------------------------------------------------
    ax1.axhline(STABLE_SELECTED, color="0.55", linewidth=0.8,
                linestyle="--", zorder=1)
    ax1.plot(x, df["selected"], "o-", color=COLOUR_A, markersize=4.0,
             linewidth=1.2, zorder=2)
    ax1.annotate("20,769", xy=(x[0], STABLE_SELECTED), xytext=(2, 6),
                 textcoords="offset points", fontsize=7, color="0.35")
    ax1.set_ylabel("Points selected", fontsize=8)

    # (b) collapsed trajectories -------------------------------------------
    ax2.plot(x, df["collapsed"], "s-", color=COLOUR_B, markersize=4.0,
             linewidth=1.2)
    ax2.set_ylabel("Degenerate trajectories", fontsize=8)
    ax2.set_xlabel(r"$n_{\mathrm{list}}$", fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{v:,}" for v in x])
        ax.tick_params(labelsize=7, width=0.6, length=2.5)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)

    ax1.set_ylim(0, max(df["selected"]) * 1.18)
    ax2.set_ylim(0, max(df["collapsed"]) * 1.25)
    ax2.set_xlabel(r"$n_{\mathrm{list}}$ (log scale)", fontsize=8)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"nlist_failure.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
