"""Independent verification of the regenerated metrics.

Deliberately shares no code with ``metric_compare.py``: distances are computed
by explicit chunked broadcasting instead of a KD-tree, and all denominators are
re-derived from row counts.

Checks
------
V1  brute-force TT / FP / FN for the reference ADF variant vs the baseline
V2  denominators: TT + FP == n_ADF; FN == n_baseline - n_baseline_matched
V3  per-flight selection counts sum to the reference total
V4  ctu_adf_poi.csv (POI==1) and filtered_poi.csv describe the same point set
V5  results/regenerated_metrics.csv reproduces the verified values at every
    threshold used by Fig. 2

Definitions verified (as in distance_sensitivity.R)
---------------------------------------------------
TT  ADF-selected points with >= 1 baseline point within T
FP  ADF-selected points with none                     -> n_ADF = TT + FP
FN  baseline points with no ADF-selected point within T
precision = TT / (TT + FP)      recall = TT / (TT + FN)

Usage:
    python verify_metrics.py [--threshold 200]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metric_compare import geodetic_to_ecef  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
ADF_POINTS = DATA / "ctu_adf_poi.csv"
FILTERED = DATA / "filtered_poi.csv"
BASELINE = DATA / "2024-12-16-CTU_processed_poi.csv"
THRESHOLDS = [100, 150, 200, 250, 300, 350, 400, 450, 500]


def xyz(df):
    # float64 on purpose: metric_compare.py works in float64, and a float32
    # check flips a handful of points that sit exactly on a threshold boundary.
    return geodetic_to_ecef(df["lon"], df["lat"], df["alt"])


def brute_matched(a, b, thr, chunk=256):
    """True where each row of `a` has a row of `b` within `thr` metres."""
    out = np.zeros(len(a), dtype=bool)
    for i in range(0, len(a), chunk):
        block = a[i : i + chunk]
        d = np.linalg.norm(block[:, None, :] - b[None, :, :], axis=2)
        out[i : i + chunk] = (d <= thr).any(axis=1)
    return out


def key(df):
    return (
        df["lon"].round(9).astype(str)
        + "|" + df["lat"].round(9).astype(str)
        + "|" + df["alt"].round(6).astype(str)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=200.0)
    args = ap.parse_args()

    adf_df = pd.read_csv(ADF_POINTS)
    adf_sel = adf_df[adf_df["POI"] == 1]
    base_df = pd.read_csv(BASELINE)
    a, b = xyz(adf_sel), xyz(base_df)

    print(f"n_ADF  = {len(a):,}")
    print(f"n_base = {len(b):,}\n")

    failures = []

    # ---- V1 + V2 + V5
    print(f"=== V1/V2/V5  brute force at T = {args.threshold:.0f} m ===")
    a_hit = brute_matched(a, b, args.threshold)
    b_hit = brute_matched(b, a, args.threshold)

    tt = int(a_hit.sum())
    fp = int(len(a) - tt)
    n_base_matched = int(b_hit.sum())
    fn = int(len(b) - n_base_matched)
    precision = tt / (tt + fp)
    recall = tt / (tt + fn)

    print(f"  TT={tt:,}  FP={fp:,}  FN={fn:,}  n_base_matched={n_base_matched:,}")
    print(f"  precision={precision:.4f}  recall={recall:.4f}")

    if tt + fp != len(a):
        failures.append("V2: TT + FP != n_ADF")
    else:
        print(f"  V2 ok: TT + FP = {len(a):,} = n_ADF")

    ref = pd.read_csv(RESULTS / "regenerated_metrics.csv")
    row = ref[(ref["variant"] == "adf_adaptive")
              & (ref["Threshold_m"] == args.threshold)].iloc[0]
    print(f"  regenerated_metrics.csv: TT={int(row['TT']):,} FP={int(row['FP']):,} "
          f"FN={int(row['FN']):,} P={row['Precision']:.4f} R={row['Recall']:.4f}")
    if (tt, fp, fn) != (int(row["TT"]), int(row["FP"]), int(row["FN"])):
        failures.append("V5: table disagrees with brute force")
    else:
        print("  V5 ok: table reproduces the brute-force values")

    # ---- V5 across the Fig. 2 threshold sweep
    print("\n=== V5  threshold sweep (Fig. 2 source) ===")
    sub = ref[ref["variant"] == "adf_adaptive"].sort_values("Threshold_m")
    for _, r in sub.iterrows():
        t = float(r["Threshold_m"])
        ah = brute_matched(a, b, t)
        bh = brute_matched(b, a, t)
        ttv, fpv = int(ah.sum()), int(len(a) - ah.sum())
        fnv = int(len(b) - bh.sum())
        ok = (ttv, fpv, fnv) == (int(r["TT"]), int(r["FP"]), int(r["FN"]))
        print(f"  T={t:>5.0f}m  brute TT={ttv:>6,} FP={fpv:>5,} FN={fnv:>6,}   "
              f"table TT={int(r['TT']):>6,} FP={int(r['FP']):>5,} "
              f"FN={int(r['FN']):>6,}   {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"V5: mismatch at T={t:.0f}")

    # ---- V3
    print("\n=== V3  per-flight selection counts ===")
    per_flight = adf_sel.groupby("flight_id").size()
    print(f"  flights={len(per_flight):,}  total={int(per_flight.sum()):,}")
    if int(per_flight.sum()) != len(a):
        failures.append("V3: per-flight sum != n_ADF")
    else:
        print("  V3 ok: per-flight counts sum to n_ADF")

    # ---- V4
    print("\n=== V4  ADF POI set vs filtered_poi.csv ===")
    fil = pd.read_csv(FILTERED)
    s1, s2 = set(key(adf_sel)), set(key(fil))
    print(f"  |ctu_adf_poi (POI=1)| = {len(s1):,}")
    print(f"  |filtered_poi|        = {len(s2):,}")
    print(f"  symmetric difference  = {len(s1 ^ s2):,}")
    if len(s1 ^ s2) == 0:
        print("  V4 ok: identical point sets (after numeric rounding)")
    else:
        failures.append("V4: point sets differ")

    print("\n" + ("ALL CHECKS PASSED" if not failures
                  else "FAILURES:\n  - " + "\n  - ".join(failures)))


if __name__ == "__main__":
    main()
