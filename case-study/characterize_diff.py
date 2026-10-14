"""Characterise the trajectory points where ADF and kNN-density disagree.

ADF and kNN-density select the same trajectory points for ~92% of the
evaluation set, which is why their aggregate metrics tie. This script
isolates the ~8% disagreement and tests the hypothesis that ADF's score
weighting pulls in points that are *less dense* -- i.e. points that pure
density (kNN) would not select.

If the ADF-only picks are significantly less dense than the kNN-only picks,
that is direct evidence that the score modulates selection beyond geometry,
and that the aggregate metric is simply too coarse to show it.

Usage:
    python characterize_diff.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from metric_compare import geodetic_to_ecef  # noqa: E402

DATA = HERE / "data"
REFERENCE = DATA / "poi_without_ctu.csv"
ADF_CSV = DATA / "ctu_adf_poi.csv"
KNN_CSV = DATA / "ctu_knn_poi_25.csv"
BASELINE = DATA / "2024-12-16-CTU_processed_poi.csv"

K_DENSITY = 100


def _key(df):
    return (
        df["flight_id"].astype(str)
        + "|" + df["lon"].round(9).astype(str)
        + "|" + df["lat"].round(9).astype(str)
        + "|" + df["alt"].round(6).astype(str)
    )


def mean_density(points_xyz, tree, k=K_DENSITY):
    """Mean distance to the k nearest reference POIs (larger = sparser)."""
    dists, _ = tree.query(points_xyz, k=k)
    return dists.mean(axis=1)


def main():
    ref = pd.read_csv(REFERENCE)
    ref_xyz = geodetic_to_ecef(ref["lon"], ref["lat"], ref["alt"]).astype("float32")
    tree = cKDTree(ref_xyz)

    adf = pd.read_csv(ADF_CSV)
    adf = adf[adf["POI"] == 1].copy()
    knn = pd.read_csv(KNN_CSV).copy()

    adf["_key"] = _key(adf)
    knn["_key"] = _key(knn)

    set_adf = set(adf["_key"])
    set_knn = set(knn["_key"])

    only_adf = adf[~adf["_key"].isin(set_knn)]
    only_knn = knn[~knn["_key"].isin(set_adf)]
    shared = len(set_adf & set_knn)

    print("=== ADF vs kNN-density selection ===")
    print(f"  |ADF|        = {len(set_adf):,}")
    print(f"  |KNN|        = {len(set_knn):,}")
    print(f"  shared       = {shared:,}  ({shared / len(set_adf):.2%} of ADF)")
    print(f"  ADF-only     = {len(only_adf):,}")
    print(f"  KNN-only     = {len(only_knn):,}")

    a_xyz = geodetic_to_ecef(only_adf["lon"], only_adf["lat"], only_adf["alt"])
    b_xyz = geodetic_to_ecef(only_knn["lon"], only_knn["lat"], only_knn["alt"])

    d_adf = mean_density(a_xyz.astype("float32"), tree)
    d_knn = mean_density(b_xyz.astype("float32"), tree)

    print(f"\n=== local density of the disagreeing points ===")
    print(f"  mean distance to {K_DENSITY} nearest reference POIs (m)")
    print(f"    ADF-only : mean {d_adf.mean():9.1f}   median {np.median(d_adf):9.1f}")
    print(f"    KNN-only : mean {d_knn.mean():9.1f}   median {np.median(d_knn):9.1f}")

    u = mannwhitneyu(d_adf, d_knn, alternative="greater")
    print(f"\n  Mann-Whitney U (ADF-only sparser than KNN-only):")
    print(f"    statistic={u.statistic:,.0f}  p={u.pvalue:.3e}")

    ratio = d_adf.mean() / d_knn.mean()
    print(f"    mean ratio ADF-only / KNN-only = {ratio:.3f}")

    if u.pvalue < 0.05 and ratio > 1.0:
        print("\n  => ADF-only picks are significantly SPARSER than KNN-only picks.")
        print("     The score is selecting low-density points that pure density misses.")
    else:
        print("\n  => no significant sparsity difference; score effect not demonstrated.")

    # ---- are the disagreeing points errors, or detections the baseline missed?
    base = pd.read_csv(BASELINE)
    base_xyz = geodetic_to_ecef(base["lon"], base["lat"], base["alt"]).astype("float32")
    base_tree = cKDTree(base_xyz)

    print("\n=== distance from the disagreeing points to the kinematic baseline ===")
    for label, pts in (("ADF-only", a_xyz), ("KNN-only", b_xyz)):
        d, _ = base_tree.query(pts.astype("float32"), k=1)
        within = "   ".join(
            f"<={r:>4}m {(d <= r).mean():6.1%}" for r in (500, 1000, 2000, 5000)
        )
        print(f"  {label:<9} median {np.median(d):8.1f} m   {within}")


if __name__ == "__main__":
    main()
