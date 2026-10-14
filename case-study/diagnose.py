"""Diagnose whether ADF's score/bandwidth design changes *what* is selected.

Answers the two questions that decide the method direction:

  Q1. Is the reference score redundant with local geometry?
      If corr(score, local density) is strong, then score-modulation is
      largely a re-parameterisation of density -- which would explain why ADF
      ties plain KNN on the current benchmark.

  Q2. Do the different ADF variants select the *same* trajectory points?
      Mask overlap |A n B| / |A| between adaptive / fixed-sigma / brute-force /
      KNN POI sets.
        high overlap (~0.95) -> the evaluation is the bottleneck
        low  overlap (~0.6)  -> the operator genuinely selects differently

Usage:
    python diagnose.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from metric_compare import geodetic_to_ecef  # noqa: E402

DATA = HERE / "data"
REFERENCE = DATA / "poi_without_ctu.csv"

RADIUS_M = 500.0
K_DENSITY = 32
SAMPLE = 100_000
SEED = 0

MASKS = {
    "adf_adaptive": "ctu_adf_poi.csv",
    "adf_fixed_500": "ctu_adf_poi_fixed_500.csv",
    "brute_force": "ctu_brutal_poi.csv",
    "knn_25": "ctu_knn_poi_25.csv",
}


def _spearman(a, b):
    res = spearmanr(a, b)
    return float(getattr(res, "statistic", res[0]))


def score_vs_density():
    df = pd.read_csv(REFERENCE)
    xyz = geodetic_to_ecef(df["lon"], df["lat"], df["alt"]).astype("float32")
    scores = df["poi_score"].to_numpy(dtype=float)

    tree = cKDTree(xyz)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(xyz), size=min(SAMPLE, len(xyz)), replace=False)
    sample = xyz[idx]

    counts = tree.query_ball_point(sample, r=RADIUS_M, return_length=True)
    dists, _ = tree.query(sample, k=K_DENSITY)
    mean_k = dists[:, 1:].mean(axis=1)  # drop self-distance

    sp_count = _spearman(scores[idx], counts)
    sp_dist = _spearman(scores[idx], mean_k)

    print("=== Q1: score vs geometry ===")
    print(f"  reference POIs          : {len(df):,} (sampled {len(idx):,})")
    print(f"  corr(score, n<{RADIUS_M:.0f}m)     : spearman {sp_count:+.4f}")
    print(f"  corr(score, mean dist {K_DENSITY}nn): spearman {sp_dist:+.4f}")
    print("  -> strong positive on count / negative on distance")
    print("     means 'high score' already implies 'high density'.")


def _key(df):
    return (
        df["flight_id"].astype(str)
        + "|" + df["lon"].round(9).astype(str)
        + "|" + df["lat"].round(9).astype(str)
        + "|" + df["alt"].round(6).astype(str)
    )


def mask_overlap():
    print("\n=== Q2: POI mask overlap ===")
    sets = {}
    for label, fname in MASKS.items():
        path = DATA / fname
        if not path.exists():
            print(f"  ! missing {fname}")
            continue
        df = pd.read_csv(path)
        if "POI" in df.columns:
            df = df[df["POI"] == 1]
        sets[label] = set(_key(df))
        print(f"  {label:<16} n={len(sets[label]):,}")

    base = "adf_adaptive"
    if base not in sets:
        return
    ref = sets[base]
    print(f"\n  overlap against '{base}' (|A n B| / |A|):")
    for label, other in sets.items():
        if label == base:
            continue
        inter = len(ref & other)
        print(
            f"    {label:<16} jaccard={inter / len(ref | other):.4f}  "
            f"|A n B|/|A|={inter / len(ref):.4f}  "
            f"|A n B|/|B|={inter / len(other):.4f}"
        )


def main():
    score_vs_density()
    mask_overlap()


if __name__ == "__main__":
    main()
