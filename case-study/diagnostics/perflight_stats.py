"""Per-flight paired comparison of ADF vs KNN density selection (existing data only).

Unit of analysis
----------------
One observation per flight (trajectory). For each flight f:

    TT_f   selected points of f with a baseline point of f within T
    P_f    TT_f / n_selected_f
    FN_f   baseline points of f with no selected point of f within T
    R_f    TT_f / (TT_f + FN_f)
    F1_f   harmonic mean of P_f and R_f

Matching is restricted to the same flight: the baseline POIs are partitioned by
flight_id, so each observation is self-contained.

Statistics
----------
Paired difference d_f = metric_ADF(f) - metric_KNN(f), over flights:

  * mean and SD of d
  * 95% percentile bootstrap CI for the mean (10,000 flight resamples)
  * Cohen's d_z = mean(d) / sd(d)   (paired standardised effect)
  * Wilcoxon signed-rank test (two-sided)

Dependence assumptions
----------------------
Flights are NOT independent samples: they share one reference POI set, overlap
spatially, and share the baseline detector. Points within a flight are
correlated, which is why the per-flight aggregate is used as the unit. Neither
device removes the shared-reference dependence, so the CI is optimistically
narrow and the test should be read as descriptive, not confirmatory.

Usage:
    python diagnostics/perflight_stats.py [--threshold 200]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
if str(CASE) not in sys.path:
    sys.path.insert(0, str(CASE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from metric_compare import geodetic_to_ecef  # noqa: E402

DATA = CASE / "data"
ADF = DATA / "ctu_adf_poi.csv"
KNN = DATA / "ctu_knn_poi_25.csv"
BASELINE = DATA / "2024-12-16-CTU_processed_poi.csv"
N_BOOT = 10_000
SEED = 0


def load_selected(path):
    df = pd.read_csv(path)
    if "POI" in df.columns:
        df = df[df["POI"] == 1]
    return df


def per_flight_metrics(sel, base_by_flight, threshold):
    rows = []
    for fid, g in sel.groupby("flight_id"):
        b = base_by_flight.get(fid)
        if b is None or len(b) == 0:
            continue
        s_xyz = geodetic_to_ecef(g["lon"], g["lat"], g["alt"])
        b_xyz = b
        b_tree = cKDTree(b_xyz)
        s_tree = cKDTree(s_xyz)

        tt = int((b_tree.query(s_xyz, k=1)[0] <= threshold).sum())
        n_sel = len(s_xyz)
        fn = int((s_tree.query(b_xyz, k=1)[0] > threshold).sum())

        p = tt / n_sel if n_sel else np.nan
        r = tt / (tt + fn) if (tt + fn) else np.nan
        f1 = 2 * p * r / (p + r) if (p + r) else np.nan
        rows.append(dict(flight_id=fid, n_sel=n_sel, n_base=len(b),
                         TT=tt, FN=fn, Precision=p, Recall=r, F1=f1))
    return pd.DataFrame(rows).set_index("flight_id")


def report(name, d, rng):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    dz = mean / sd if sd else float("nan")
    try:
        p_value = float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        p_value = float("nan")
    print(f"  {name:<10} mean Δ={mean:+.4f}  sd={sd:.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  d_z={dz:+.3f}  "
          f"Wilcoxon p={p_value:.3f}")
    return mean, lo, hi, dz, p_value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=200.0)
    args = ap.parse_args()

    base = pd.read_csv(BASELINE)
    print(f"baseline columns: {list(base.columns)}")
    if "flight_id" not in base.columns:
        raise SystemExit("baseline has no flight_id; per-flight pairing not possible")
    base_by_flight = {
        fid: geodetic_to_ecef(g["lon"], g["lat"], g["alt"])
        for fid, g in base.groupby("flight_id")
    }

    a = per_flight_metrics(load_selected(ADF), base_by_flight, args.threshold)
    k = per_flight_metrics(load_selected(KNN), base_by_flight, args.threshold)
    common = a.index.intersection(k.index)
    a, k = a.loc[common], k.loc[common]

    print(f"\nflights with both sets and >=1 baseline point: {len(common):,}")
    print(f"T = {args.threshold:.0f} m\n")
    print(f"  {'metric':<10}{'ADF mean':>10}{'KNN mean':>10}")
    for m in ("Precision", "Recall", "F1"):
        print(f"  {m:<10}{a[m].mean():>10.4f}{k[m].mean():>10.4f}")

    print("\npaired per-flight differences (ADF − KNN):")
    rng = np.random.default_rng(SEED)
    for m in ("Precision", "Recall", "F1"):
        report(m, (a[m] - k[m]).to_numpy(), rng)

    print("\n  interpretation: a CI spanning 0 means the difference is not")
    print("  distinguishable from zero at this sample size; no superiority is")
    print("  claimed. Flights share a reference set and detector, so intervals")
    print("  are optimistically narrow (descriptive, not confirmatory).")


if __name__ == "__main__":
    main()
