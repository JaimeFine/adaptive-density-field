"""Regenerate Fig. 3 (spatial distribution of the extracted POIs).

Why this script exists
----------------------
The figure currently in the manuscript (``thesis-v5/baseline.pdf``) was
produced on 2026-01-28 by the January draft pipeline
(``drafts/v2-2026-01-28``), i.e. *before* the ENU coordinate-conversion
correction and the other fixes applied on 2026-10-09. It is byte-identical to
the copy submitted in February, and no script in the current harness
reproduced it, so it could not be regenerated as it stood. This script
supersedes it.

Sources of truth
----------------
* ADF detections     -- ``results/poi_sets/nlist_4096.csv``
                        (canonical configuration: nlist=4096, nprobe=16, k=100)
* reference labels   -- ``data/2024-12-16-CTU_processed_poi.csv``
                        (20,501 kinematic labels from the motion-residual pipeline)

Concordance uses the same rule as the rest of the harness (``metric_compare.py``:
WGS84 geodetic -> ECEF, KD-tree radius query, 200 m tolerance by default), so
the precision/recall printed below must agree with Table 1 and Figure 2. They
are printed for exactly that check; nothing is recomputed by a second method.

Outputs
-------
``results/poi_map.png`` and ``results/poi_map.pdf``

Deliberately written to ``results/`` rather than over ``thesis-v5/`` so that
replacing the manuscript figure remains an explicit, reviewable decision.

Usage:
    python poi_map_plot.py [--threshold 200] [--span 2.4]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import metric_compare as mc  # noqa: E402

HERE = Path(__file__).resolve().parent
ADF_SET = HERE / "results" / "poi_sets" / "nlist_4096.csv"
BASE_SET = HERE / "data" / "2024-12-16-CTU_processed_poi.csv"
OUT = HERE / "results"

COLOUR_MATCHED = "#12406b"   # both methods flag the location
COLOUR_ADF_ONLY = "#e08214"  # ADF only
COLOUR_BASE_ONLY = "#9a9a9a"  # reference labels only


def densest_centre(lon: np.ndarray, lat: np.ndarray, cell: float = 0.5):
    """Centre of the densest ``cell``-degree box, as the zoom window anchor."""
    frame = pd.DataFrame(
        {"cl": np.floor(lon / cell) * cell, "ct": np.floor(lat / cell) * cell}
    )
    cl, ct = frame.groupby(["cl", "ct"]).size().idxmax()
    return cl + cell / 2.0, ct + cell / 2.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=200.0,
                    help="matching tolerance in metres (default 200)")
    ap.add_argument("--span-lon", type=float, default=3.4,
                    help="zoom window width in degrees of longitude (default 3.4)")
    ap.add_argument("--span-lat", type=float, default=2.0,
                    help="zoom window height in degrees of latitude (default 2.0)")
    args = ap.parse_args()

    if not ADF_SET.exists():
        raise SystemExit(f"{ADF_SET} not found")
    if not BASE_SET.exists():
        raise SystemExit(f"{BASE_SET} not found")

    adf = pd.read_csv(ADF_SET)
    base = pd.read_csv(BASE_SET)
    print(f"ADF detections   : {len(adf):,}  ({ADF_SET.name})")
    print(f"reference labels : {len(base):,}  ({BASE_SET.name})")

    # --- concordance, using the harness's own metric ----------------------
    adf_xyz = mc.geodetic_to_ecef(adf["lon"], adf["lat"], adf["alt"])
    base_xyz = mc.geodetic_to_ecef(base["lon"], base["lat"], base["alt"])

    base_tree = cKDTree(base_xyz)
    adf_tree = cKDTree(adf_xyz)

    adf_hit = base_tree.query_ball_point(
        adf_xyz, r=args.threshold, return_length=True) > 0
    base_hit = adf_tree.query_ball_point(
        base_xyz, r=args.threshold, return_length=True) > 0

    tt = int(adf_hit.sum())
    fp = len(adf_hit) - tt
    fn = int((~base_hit).sum())
    precision = tt / (tt + fp) if (tt + fp) else 0.0
    recall = tt / (tt + fn) if (tt + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    print(f"\nmatching tolerance: {args.threshold:g} m")
    print(f"  matched (TT)   : {tt:,}")
    print(f"  ADF only (FP)  : {fp:,}")
    print(f"  labels only(FN): {fn:,}")
    print(f"  precision      : {precision:.4f}")
    print(f"  recall         : {recall:.4f}")
    print(f"  F1             : {f1:.4f}")
    print("  (these must agree with Table 1 and Figure 2)")

    # --- zoom window ------------------------------------------------------
    lon_c, lat_c = densest_centre(
        np.concatenate([adf["lon"].to_numpy(), base["lon"].to_numpy()]),
        np.concatenate([adf["lat"].to_numpy(), base["lat"].to_numpy()]),
    )
    half_lon = args.span_lon / 2.0
    half_lat = args.span_lat / 2.0
    x0, x1 = lon_c - half_lon, lon_c + half_lon
    y0, y1 = lat_c - half_lat, lat_c + half_lat
    print(f"\nzoom window      : lon [{x0:.2f}, {x1:.2f}]  "
          f"lat [{y0:.2f}, {y1:.2f}]  "
          f"({args.span_lon:g} x {args.span_lat:g} deg)")

    def in_window(lon: pd.Series, lat: pd.Series) -> np.ndarray:
        return ((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)).to_numpy()

    adf_in = in_window(adf["lon"], adf["lat"])
    base_in = in_window(base["lon"], base["lat"])
    print(f"points in window : ADF {adf_in.sum():,}  labels {base_in.sum():,}")

    # --- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Draw the two "only" classes first, the agreement on top, so the
    # concordance is the visually dominant layer.
    ax.scatter(
        base["lon"].to_numpy()[base_in & ~base_hit],
        base["lat"].to_numpy()[base_in & ~base_hit],
        s=2.2, c=COLOUR_BASE_ONLY, alpha=0.55, linewidths=0,
        rasterized=True, label="Reference labels only",
    )
    ax.scatter(
        adf["lon"].to_numpy()[adf_in & ~adf_hit],
        adf["lat"].to_numpy()[adf_in & ~adf_hit],
        s=2.2, c=COLOUR_ADF_ONLY, alpha=0.80, linewidths=0,
        rasterized=True, label="ADF only",
    )
    ax.scatter(
        adf["lon"].to_numpy()[adf_in & adf_hit],
        adf["lat"].to_numpy()[adf_in & adf_hit],
        s=1.8, c=COLOUR_MATCHED, alpha=0.55, linewidths=0,
        rasterized=True, label=f"Both, within {args.threshold:g} m",
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    # keep degrees square at this latitude so the map is not distorted
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat_c)))

    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.tick_params(labelsize=8, width=0.6, length=3.0)
    ax.ticklabel_format(useOffset=False)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.22, linewidth=0.5)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)

    leg = ax.legend(
        loc="lower right", fontsize=7.5, markerscale=3.5,
        frameon=True, framealpha=0.95, borderpad=0.4, handletextpad=0.4,
    )
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"poi_map.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
