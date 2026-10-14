"""Density-stratified concordance: does ADF beat kNN-density where density fails?

Motivation
----------
The aggregate 150-300 m matching metric is dominated by the ~92% of trajectory
points that ADF and kNN-density select identically, so it cannot show where the
two methods differ. This script re-evaluates both methods *within* strata of
local reference density, so behaviour in sparse regions is measured directly.

Definitions
-----------
d1(p)         distance (metres) to the nearest reference POI -- the sparsity
              proxy. A continuous measure is used because a fixed-radius count
              is degenerate here: over half of the evaluation points have zero
              reference POIs within 500 m, which empties the lower strata.
S1..S5        quintiles of d1 over the evaluation set
              (S1 = densest, S5 = sparsest: larger d1 means fewer nearby POIs)
matched       a selected point with >= 1 kinematic-baseline point within T
              metres (3D ECEF), and symmetrically for baseline points
precision_s   TT_s / (TT_s + FP_s)      within stratum s
recall_s      TT_s / (TT_s + FN_s)      within stratum s
              (FN counted over baseline points assigned to s)

Fairness / validity checks printed by this script
-------------------------------------------------
* both methods must select the same number of points (cardinality parity);
* the sum of selected points over strata must equal the unstratified total;
* the pooled precision must reproduce the unstratified value.
d1 is the *nearest-neighbour* distance -- a different functional form from
kNN-density's mean distance to k = 25 neighbours -- and it uses no kernel
bandwidth, so it is not the selection rule of either method. It is still a
geometric quantity and therefore correlates with both scores, so the
stratified result shows whether the gap grows with sparsity, not that either
method is independent of geometry.

Usage:
    python sparse_metric.py [--threshold 2

Usage:
    python sparse_metric.py [--threshold 200] [--radii 500 1000 2000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metric_compare import geodetic_to_ecef  # noqa: E402

DATA = HERE / "data"
REFERENCE = DATA / "poi_without_ctu.csv"
EVAL_POINTS = DATA / "ctu_adf_poi.csv"
METHODS = {
    "adf_adaptive": (EVAL_POINTS, "POI"),
    "knn_25": (DATA / "ctu_knn_poi_25.csv", None),
}
BASELINE = DATA / "2024-12-16-CTU_processed_poi.csv"

N_STRATA = 5

STRATUM_LABELS = (
    ["S1 densest"] + [f"S{i + 1}" for i in range(1, N_STRATA - 1)] + [f"S{N_STRATA} sparsest"]
)


def to_xyz(df):
    return geodetic_to_ecef(df["lon"], df["lat"], df["alt"]).astype("float32")


def rho_of(pts_xyz, ref_tree, radius):
    """Count reference POIs within `radius` metres of each point."""
    return ref_tree.query_ball_point(pts_xyz, r=radius, return_length=True)


def nearest_ref_dist(pts_xyz, ref_tree):
    """Distance (m) to the nearest reference POI -- a continuous sparsity proxy.

    Preferred over a fixed-radius count here because more than half of the
    evaluation points have zero reference POIs within 500 m, which collapses
    the lower quantiles of that count and leaves empty strata.
    """
    dist, _ = ref_tree.query(pts_xyz, k=1)
    return dist


def matched_within(pts_xyz, other_tree, threshold):
    """True where the nearest point of `other_tree` is within `threshold`."""
    dist, _ = other_tree.query(pts_xyz, k=1)
    return dist <= threshold


def evaluate(threshold, ref_tree, eval_df, base_df, base_xyz):
    sparsity_eval = nearest_ref_dist(to_xyz(eval_df), ref_tree)
    edges = np.quantile(sparsity_eval, np.linspace(0.0, 1.0, N_STRATA + 1))
    inner = edges[1:-1]

    def stratum_of(values):
        return np.clip(np.searchsorted(inner, values, side="right"), 0, N_STRATA - 1)

    base_strata = stratum_of(nearest_ref_dist(base_xyz, ref_tree))
    base_tree = cKDTree(base_xyz)

    print(f"\n{'=' * 78}")
    print(f"T = {threshold:.0f} m   strata = {N_STRATA} "
          f"(by distance to nearest reference POI)")
    print(f"  edges (m) = {[round(e) for e in edges]}")
    print(f"{'=' * 78}")

    rows = []
    for name, (path, poi_col) in METHODS.items():
        df = pd.read_csv(path)
        if poi_col is not None and poi_col in df.columns:
            df = df[df[poi_col] == 1]
        sel_xyz = to_xyz(df)
        sel_strata = stratum_of(nearest_ref_dist(sel_xyz, ref_tree))
        sel_tree = cKDTree(sel_xyz)
        sel_matched = matched_within(sel_xyz, base_tree, threshold)
        base_matched = matched_within(base_xyz, sel_tree, threshold)

        n_total = len(sel_xyz)
        for s in range(len(inner) + 1):
            m = sel_strata == s
            tt = int(sel_matched[m].sum())
            fp = int(m.sum() - tt)
            b = base_strata == s
            fn = int((~base_matched[b]).sum())
            precision = tt / (tt + fp) if (tt + fp) else float("nan")
            recall = tt / (tt + fn) if (tt + fn) else float("nan")
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) and not np.isnan(precision + recall)
                else float("nan")
            )
            rows.append(
                dict(method=name, stratum=STRATUM_LABELS[s], n_selected=m.sum(),
                     n_baseline=int(b.sum()), TT=tt, FP=fp, FN=fn,
                     Precision=precision, Recall=recall, F1=f1)
            )

        pooled_tt = int(sel_matched.sum())
        pooled_prec = pooled_tt / n_total
        print(f"\n  [{name}]  selected={n_total:,}  "
              f"pooled precision={pooled_prec:.4f}  "
              f"(validity: sum over strata = {sum(r['n_selected'] for r in rows if r['method'] == name):,})")

    table = pd.DataFrame(rows)
    wide = table.pivot(index="stratum", columns="method",
                       values=["Precision", "Recall", "F1"]).round(4)
    print("\n" + wide.to_string())

    print("\n  support (n_selected / n_baseline) per stratum:")
    support = table.pivot(index="stratum", columns="method",
                          values=["n_selected", "n_baseline"])
    print(support.to_string())
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=200.0)
    args = ap.parse_args()

    ref = pd.read_csv(REFERENCE)
    ref_tree = cKDTree(to_xyz(ref))
    eval_df = pd.read_csv(EVAL_POINTS)
    base_df = pd.read_csv(BASELINE)
    base_xyz = to_xyz(base_df)

    print(f"reference POIs : {len(ref):,}")
    print(f"evaluation pts : {len(eval_df):,}")
    print(f"baseline POIs  : {len(base_df):,}")
    print(f"threshold      : {args.threshold:.0f} m")

    table = evaluate(args.threshold, ref_tree, eval_df, base_df, base_xyz)

    out = HERE / "results" / "sparse_stratified.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
