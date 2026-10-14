"""Paired per-flight scatter: ADF against the plain k-NN density scan.

The figure exists to make the manuscript's null result visible. If the two
methods selected the same points, every trajectory would sit on the diagonal;
the point is that they nearly do, while a few do not.

Reuse, not reimplementation
---------------------------
The per-flight rule is imported from ``diagnostics/perflight_stats.py``,
which is the module that produced the numbers quoted in
Section 5.3: one observation per trajectory, matching restricted to the same
flight, 200 m tolerance in 3D ECEF, and a percentile bootstrap over flights.
Nothing is recomputed by a second method here.

Caveat carried from the source module
-------------------------------------
Flights share one reference set and one label detector, so the interval is
optimistically narrow and is descriptive rather than confirmatory.

Outputs
-------
``results/perflight_scatter.png`` and ``results/perflight_scatter.pdf``

Usage:
    python perflight_scatter_plot.py [--threshold 200]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "diagnostics"))

import perflight_stats as pfs  # noqa: E402

OUT = HERE / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=200.0)
    args = ap.parse_args()

    base = pd.read_csv(pfs.BASELINE)
    base_by_flight = {
        fid: pfs.geodetic_to_ecef(g["lon"], g["lat"], g["alt"])
        for fid, g in base.groupby("flight_id")
    }

    a = pfs.per_flight_metrics(
        pfs.load_selected(pfs.ADF), base_by_flight, args.threshold)
    k = pfs.per_flight_metrics(
        pfs.load_selected(pfs.KNN), base_by_flight, args.threshold)

    common = a.index.intersection(k.index)
    a, k = a.loc[common], k.loc[common]

    # The harness reports Precision, Recall and F1 in that order from a single
    # generator, so the F1 bootstrap depends on the state left by the two prior
    # calls. That order is replayed here so the interval annotated on the figure
    # is the one quoted in the manuscript rather than a second draw.
    rng = np.random.default_rng(pfs.SEED)
    stats = {}
    for metric in ("Precision", "Recall", "F1"):
        d = (a[metric] - k[metric]).to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean()
                          for _ in range(pfs.N_BOOT)])
        lo_, hi_ = np.percentile(boots, [2.5, 97.5])
        stats[metric] = dict(n=len(d), mean=float(d.mean()),
                             lo=float(lo_), hi=float(hi_))

    mean = stats["F1"]["mean"]
    lo, hi = stats["F1"]["lo"], stats["F1"]["hi"]

    print(f"trajectories paired        : {len(a):,}")
    print(f"finite per-flight F1 diffs : {stats['F1']['n']:,}")
    print(f"tolerance                  : {args.threshold:g} m")
    print(f"ADF mean per-flight F1     : {a['F1'].mean():.4f}")
    print(f"k-NN scan mean per-flight F1: {k['F1'].mean():.4f}")
    print(f"mean paired difference     : {mean:+.4f}")
    print(f"95% bootstrap CI           : [{lo:+.4f}, {hi:+.4f}]")
    print(f"  (CI contains 0: {lo <= 0 <= hi}; "
          f"{pfs.N_BOOT:,} resamples, seed {pfs.SEED})")

    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    lim = 1.0
    ax.plot([0, lim], [0, lim], color="0.45", linewidth=0.8,
            linestyle="--", zorder=1)
    ax.scatter(k["F1"], a["F1"], s=9, c="#12406b", alpha=0.55,
               linewidths=0, zorder=2)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("k-NN density scan F1 (per flight)", fontsize=8)
    ax.set_ylabel("ADF F1 (per flight)", fontsize=8)
    ax.tick_params(labelsize=7, width=0.6, length=2.5)
    ax.grid(True, alpha=0.22, linewidth=0.5)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)

    ax.text(
        0.04, 0.96,
        f"mean $\\Delta$F1 = {mean:+.4f}\n95% CI [{lo:+.4f}, {hi:+.4f}]",
        transform=ax.transAxes, fontsize=7, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.75", linewidth=0.5, alpha=0.9),
    )

    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"perflight_scatter.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
