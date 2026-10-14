"""Diagnostic: is the score's influence suppressed by the amplitude/bandwidth coupling?

This is a read-only audit of EXISTING verified outputs. It computes no new
field values and changes no implementation, per the diagnostic constraints.

What it tests
-------------
H1 (decouple amplitude from bandwidth): the proposal's first-choice experiment
    is "fixed bandwidth, score retained as amplitude". That variant ALREADY
    EXISTS as ``ctu_adf_poi_fixed_500.csv`` (``bandwidth.py`` keeps the score in
    the numerator and removes ``sigma0 / s``). So H1 can be tested against
    existing data rather than re-run.

H2 (score/bandwidth interaction): contribution of a point is
        s * exp(-d^2 s^2 / (2 sigma0^2))
    which, at non-zero distance, peaks at s* = sigma0 / d and DECREASES for
    larger s. This reports where the real scores sit relative to s*.

H3 (thresholding): the per-trajectory 75th-percentile cut depends only on the
    ORDER of field values, so it cannot see magnitude changes. This reports how
    much the bandwidth change reorders the field (rank correlation) versus how
    much it changes the selection.

Channel decomposition
---------------------
    bandwidth channel = 1 - overlap(adaptive, fixed sigma)
    score channel     = 1 - overlap(adaptive, knn)

Usage:
    python diagnostics/score_suppression.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
if str(CASE) not in sys.path:
    sys.path.insert(0, str(CASE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from metric_compare import geodetic_to_ecef  # noqa: E402

DATA = CASE / "data"
SIGMA0 = 500.0
EPS = 1e-6

KEY_COLS = ["flight_id", "lon", "lat", "alt"]


def keys(df):
    return (
        df["flight_id"].astype(str)
        + "|" + df["lon"].round(9).astype(str)
        + "|" + df["lat"].round(9).astype(str)
        + "|" + df["alt"].round(6).astype(str)
    )


def overlap(a_df, b_df):
    sa, sb = set(keys(a_df)), set(keys(b_df))
    return len(sa & sb) / len(sa), len(sa), len(sa & sb)


def main():
    # ---------- Task 1: score audit ----------
    ref = pd.read_csv(DATA / "poi_without_ctu.csv")
    s = ref["poi_score"].to_numpy(dtype=float)
    print("=== Task 1: score audit (reference POI set) ===")
    print(f"  n = {len(s):,}")
    print(f"  min={s.min():.6g}  p25={np.percentile(s,25):.6g}  "
          f"median={np.median(s):.6g}  p75={np.percentile(s,75):.6g}  max={s.max():.6g}")
    print(f"  non-positive scores : {int((s <= 0).sum()):,}")
    print(f"  scores <= eps(1e-6) : {int((s <= EPS).sum()):,}")
    finite_pos = s[s > EPS]
    if len(finite_pos):
        bw = SIGMA0 / finite_pos
        print(f"  implied sigma_j = 500 / s : min={bw.min():.4g}  "
              f"median={np.median(bw):.4g}  max={bw.max():.4g} m")
    print("  amplitude channel uses s directly; bandwidth channel uses 500/s")

    # ---------- H1: amplitude-only variant already exists ----------
    adf = pd.read_csv(DATA / "ctu_adf_poi.csv")
    fix = pd.read_csv(DATA / "ctu_adf_poi_fixed_500.csv")
    knn = pd.read_csv(DATA / "ctu_knn_poi_25.csv")

    print("\n=== H1: fixed-bandwidth + score-amplitude  vs  current coupling ===")
    for label, df in (("adaptive(now)", adf), ("fixed_sigma_500", fix)):
        assert len(df) == len(adf), "row counts differ"
    same_geom = (
        (adf["flight_id"].astype(str) == fix["flight_id"].astype(str)).all()
        and np.allclose(adf["lon"], fix["lon"])
        and np.allclose(adf["lat"], fix["lat"])
        and np.allclose(adf["alt"], fix["alt"])
    )
    print(f"  geometric columns identical row-for-row: {same_geom}")

    ov_bf, n_a, n_i = overlap(adf[adf.POI == 1], fix[fix.POI == 1])
    ov_bk, _, n_k = overlap(adf[adf.POI == 1], knn)
    print(f"  |ADF| = {n_a:,}")
    print(f"  overlap(adaptive, fixed sigma) = {ov_bf:.6f}   ({n_i:,} points)")
    print(f"  overlap(adaptive, knn)         = {ov_bk:.6f}   ({n_k:,} points)")
    print(f"  -> bandwidth channel changes {100*(1-ov_bf):.2f}% of selections")
    print(f"  -> score channel     changes {100*(1-ov_bk):.2f}% of selections")

    # ---------- H3: does the bandwidth change reorder the field? ----------
    print("\n=== H3: rank correlation of the field under bandwidth change ===")
    rho_all = spearmanr(adf["ADF"], fix["ADF"]).statistic
    sel = adf["POI"] == 1
    rho_sel = spearmanr(adf.loc[sel, "ADF"], fix.loc[sel, "ADF"]).statistic
    print(f"  Spearman(adaptive field, fixed-sigma field), all points : {rho_all:.6f}")
    print(f"  Spearman(...), selected points only                     : {rho_sel:.6f}")
    print("  (a value near 1 means the bandwidth change barely reorders the field,")
    print("   so a rank threshold can hardly be sensitive to it)")

    # ---------- H2: where do real scores sit relative to s* = sigma0 / d ? ----------
    print("\n=== H2: score vs the contribution peak s* = sigma0 / d ===")
    sel_pts = adf[adf.POI == 1]
    ref_xyz = geodetic_to_ecef(ref["lon"], ref["lat"], ref["alt"])
    tree = cKDTree(ref_xyz)
    pts = geodetic_to_ecef(sel_pts["lon"], sel_pts["lat"], sel_pts["alt"])
    d1, idx1 = tree.query(pts, k=1)
    s_near = s[idx1]
    with np.errstate(divide="ignore"):
        s_star = np.where(d1 > 0, SIGMA0 / np.maximum(d1, 1e-9), np.inf)
    ratio = s_near / s_star
    print(f"  distance to nearest reference POI: median {np.median(d1):.1f} m")
    print(f"  score of nearest POI  : median {np.median(s_near):.4g}")
    print(f"  s* = 500/d            : median {np.median(s_star):.4g}")
    print(f"  ratio s/s* : p10={np.percentile(ratio,10):.4g} "
          f"median={np.median(ratio):.4g} p90={np.percentile(ratio,90):.4g}")
    above = float((ratio > 1).mean())
    print(f"  fraction of selected points with s > s* (contribution in the")
    print(f"  DECREASING regime): {above:.2%}")

    print("\n=== summary ===")
    print("  H1 was already testable and is ANSWERED by existing outputs.")
    print("  H2/H3 give the mechanism; variant C (bandwidth-only) remains unmeasured.")


if __name__ == "__main__":
    main()
