"""Paired per-flight tests for the clean component pairs, plus a
small-flight sensitivity check on the aggregation reversal.

Why this exists
---------------
The VBKDE decomposition reported so far is POOLED ONLY:

    kde_tree -> vbkde_amp1   bandwidth rule alone      -0.0026 F1
    vbkde_amp1 -> vbkde      density-consistent amplitude   +0.0076 F1

A pooled delta cannot be trusted to carry the same sign per flight: the
pooled and per-flight aggregations disagree in sign for ADF vs the k-NN
density estimator. So if any component-level claim is retained, these two
pairs must be tested on the paired per-flight scale.

Pairwise differences
--------------------
    kde_tree  vs vbkde_amp1 : same neighbour set (uncapped radius, float64),
                              same unit amplitude, same median bandwidth
                              (500 m), different bandwidth RULE
                              (fixed sigma0  vs  density-driven h_j)
    vbkde_amp1 vs vbkde     : same neighbour set, same h_j, different
                              AMPLITUDE (1  vs  1/h_j^3)
Both are therefore single-factor comparisons in implementation terms.

Small-flight sensitivity
------------------------
The per-flight mean weights every flight equally, and a flight with very few
points contributes an unstable per-flight F1 (at P75 a 2-point flight selects
1 point, so its precision is 0 or 1). This repeats the ADF vs k-NN and ADF vs
VBKDE comparisons restricted to flights with at least MIN_SEL selected points,
to test whether the reversal is driven by tiny flights.

Usage:
    python diagnostics/component_pairs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
if str(CASE) not in sys.path:
    sys.path.insert(0, str(CASE))

import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from metric_compare import geodetic_to_ecef  # noqa: E402

DATA = CASE / "data"
BASELINE = DATA / "2024-12-16-CTU_processed_poi.csv"
N_BOOT = 10_000
SEED = 0
T = 200.0
MIN_SEL = 20          # "small flight" threshold on selected points


def load_selected(fname):
    df = pd.read_csv(DATA / fname)
    if "POI" in df.columns:
        df = df[df["POI"] == 1]
    return df


def per_flight(sel, base_by_flight, threshold=T):
    rows = []
    for fid, g in sel.groupby("flight_id"):
        b = base_by_flight.get(fid)
        if b is None or len(b) == 0:
            continue
        s_xyz = geodetic_to_ecef(g["lon"], g["lat"], g["alt"])
        s_tree = cKDTree(s_xyz)
        b_tree = cKDTree(b)
        tt = int((b_tree.query(s_xyz, k=1)[0] <= threshold).sum())
        n_sel = len(s_xyz)
        fn = int((s_tree.query(b, k=1)[0] > threshold).sum())
        p = tt / n_sel if n_sel else np.nan
        r = tt / (tt + fn) if (tt + fn) else np.nan
        f1 = 2 * p * r / (p + r) if (p + r) else np.nan
        rows.append(dict(flight_id=fid, n_sel=n_sel, n_base=len(b),
                         Precision=p, Recall=r, F1=f1))
    return pd.DataFrame(rows).set_index("flight_id")


def report(label, d, rng):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    dz = mean / sd if sd else float("nan")
    try:
        pval = float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        pval = float("nan")
    span = "resolvable" if (lo > 0 or hi < 0) else "includes 0"
    print(f"  {label:<34} n={len(d):>4}  mean D={mean:+.4f}  "
          f"CI [{lo:+.4f}, {hi:+.4f}]  d_z={dz:+.3f}  p={pval:.3g}  {span}")
    return mean, lo, hi, dz, pval


def main() -> int:
    base = pd.read_csv(BASELINE)
    base_by_flight = {
        fid: geodetic_to_ecef(g["lon"], g["lat"], g["alt"])
        for fid, g in base.groupby("flight_id")
    }
    cache = {}
    for fname in ("ctu_adf_poi.csv", "ctu_adf_poi_base_vbkde.csv",
                  "ctu_adf_poi_base_vbkde_amp1.csv",
                  "ctu_adf_poi_base_kde_tree.csv",
                  "ctu_adf_poi_base_knn_kernel.csv",
                  "ctu_adf_poi_base_nw_tiesafe.csv",
                  "ctu_adf_poi_fact_neither.csv",
                  "ctu_adf_poi_fact_score_only.csv",
                  "ctu_adf_poi_fact_adaptive_only.csv",
                  "ctu_adf_poi_sigma0_100.csv",
                  "ctu_adf_poi_sigma0_250.csv",
                  "ctu_adf_poi_sigma0_1000.csv",
                  "ctu_adf_poi_sigma0_2000.csv",
                  "ctu_adf_poi_fixedbw_500.csv",
                  "ctu_adf_poi_fixedbw_1000.csv",
                  "ctu_knn_poi_25.csv",
                  # nlist sweep (nlist_sweep.py)
                  "ctu_adf_poi_nlist_64.csv",
                  "ctu_adf_poi_nlist_256.csv",
                  "ctu_adf_poi_nlist_1024.csv",
                  "ctu_adf_poi_nlist_4096.csv",
                  "ctu_adf_poi_nlist_16384.csv"):
        cache[fname] = per_flight(load_selected(fname), base_by_flight)

    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("CLEAN COMPONENT PAIRS - paired per-flight (does the pooled sign hold?)")
    print("=" * 78)
    pairs = [
        ("kde_tree -> vbkde_amp1", "bandwidth rule (fixed vs density)",
         "ctu_adf_poi_base_kde_tree.csv", "ctu_adf_poi_base_vbkde_amp1.csv"),
        ("vbkde_amp1 -> vbkde", "amplitude (1 vs 1/h^3)",
         "ctu_adf_poi_base_vbkde_amp1.csv", "ctu_adf_poi_base_vbkde.csv"),
        ("kde_tree -> vbkde", "bandwidth + amplitude",
         "ctu_adf_poi_base_kde_tree.csv", "ctu_adf_poi_base_vbkde.csv"),
        ("2x2: neither -> score_only", "amplitude only",
         "ctu_adf_poi_fact_neither.csv", "ctu_adf_poi_fact_score_only.csv"),
        ("2x2: neither -> adaptive_only", "bandwidth only",
         "ctu_adf_poi_fact_neither.csv", "ctu_adf_poi_fact_adaptive_only.csv"),
        ("2x2: neither -> both", "amplitude + bandwidth",
         "ctu_adf_poi_fact_neither.csv", "ctu_adf_poi.csv"),
    ]
    for label, note, fa, fb in pairs:
        a, b = cache[fa], cache[fb]
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        print(f"\n{' '*2}{label}   [{note}]")
        report("F1", (a["F1"] - b["F1"]).to_numpy(), rng)

    # sigma0 sweep.  NOTE the sign convention differs from the arrow pairs
    # above: here D = first - second, so a POSITIVE mean D means the FIRST
    # variant scores higher per flight.
    #   ctu_adf_poi.csv          coupled sigma0 = 500  (the paper's value)
    #   ctu_adf_poi_sigma0_<v>   coupled sigma0 = v    (the ADF operator)
    #   ctu_adf_poi_fixedbw_<v>  decoupled sigma = v   (what bandwidth.py ran)
    print("\n" + "=" * 78)
    print("SIGMA0 SWEEP - paired per-flight  (D = first - second; D>0 favours FIRST)")
    print("=" * 78)
    sweep_pairs = [
        ("sigma0_250 vs sigma0_500", "coupled, narrower scale",
         "ctu_adf_poi_sigma0_250.csv", "ctu_adf_poi.csv"),
        ("sigma0_100 vs sigma0_500", "coupled, much narrower scale",
         "ctu_adf_poi_sigma0_100.csv", "ctu_adf_poi.csv"),
        ("sigma0_1000 vs sigma0_500", "coupled, wider scale",
         "ctu_adf_poi_sigma0_1000.csv", "ctu_adf_poi.csv"),
        ("sigma0_2000 vs sigma0_500", "coupled, much wider scale",
         "ctu_adf_poi_sigma0_2000.csv", "ctu_adf_poi.csv"),
        ("sigma0_250 vs fixedbw_500", "best coupled vs decoupled at 500",
         "ctu_adf_poi_sigma0_250.csv", "ctu_adf_poi_fixedbw_500.csv"),
        ("fixedbw_1000 vs fixedbw_500", "decoupled, wider scale",
         "ctu_adf_poi_fixedbw_1000.csv", "ctu_adf_poi_fixedbw_500.csv"),
    ]
    for label, note, fa, fb in sweep_pairs:
        a, b = cache[fa], cache[fb]
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        print(f"\n{' '*2}{label}   [{note}]")
        report("F1", (a["F1"] - b["F1"]).to_numpy(), rng)

    print("\n" + "=" * 78)
    print("NLIST SWEEP - paired per-flight  (D = first - second; D>0 favours FIRST)")
    print("=" * 78)
    # Parity gate: nlist_sweep.py reproduced ctu_adf_poi.csv exactly at
    # nlist=4096, so the first row below must be an exact zero difference on
    # every flight.  It is carried as a self-check on this harness.
    # At fixed nprobe=16 the retrieval is bit-identical for nlist <= 4096, so
    # every contrast inside that range is expected to be exactly 0 too.  The
    # only informative contrast is nlist=16384, where nprobe=16 covers a far
    # smaller share of the index and the selection changes.
    nlist_pairs = [
        ("nlist_4096 vs adf_adaptive", "parity self-check",
         "ctu_adf_poi_nlist_4096.csv", "ctu_adf_poi.csv"),
        ("nlist_64 vs nlist_4096", "coarse partition",
         "ctu_adf_poi_nlist_64.csv", "ctu_adf_poi_nlist_4096.csv"),
        ("nlist_256 vs nlist_4096", "coarse partition",
         "ctu_adf_poi_nlist_256.csv", "ctu_adf_poi_nlist_4096.csv"),
        ("nlist_1024 vs nlist_4096", "coarse partition",
         "ctu_adf_poi_nlist_1024.csv", "ctu_adf_poi_nlist_4096.csv"),
        ("nlist_16384 vs nlist_4096", "finer than published",
         "ctu_adf_poi_nlist_16384.csv", "ctu_adf_poi_nlist_4096.csv"),
    ]
    for label, note, fa, fb in nlist_pairs:
        a, b = cache[fa], cache[fb]
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        print(f"\n{' '*2}{label}   [{note}]")
        report("F1", (a["F1"] - b["F1"]).to_numpy(), rng)

    # The divergent value trades precision for recall; report both explicitly
    # so the trade is not hidden behind a single F1 difference.
    a = cache["ctu_adf_poi_nlist_16384.csv"]
    b = cache["ctu_adf_poi_nlist_4096.csv"]
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    print("\n  nlist_16384 vs nlist_4096 -- the other two metrics")
    report("Precision", (a["Precision"] - b["Precision"]).to_numpy(), rng)
    report("Recall", (a["Recall"] - b["Recall"]).to_numpy(), rng)

    print("\n" + "=" * 78)
    print(f"SMALL-FLIGHT SENSITIVITY (restrict to flights with n_sel >= {MIN_SEL})")
    print("=" * 78)
    # NOTE: two distinct k-NN comparators exist and must NOT be conflated.
    #   ctu_knn_poi_25.csv               local-density SCAN: keep the 25% of
    #                                    points with the smallest mean distance
    #                                    to their k nearest neighbours
    #                                    (knn_processor.py).  This is the
    #                                    baseline the manuscript's k-NN passage
    #                                    quotes (75.69% precision / 66.18% F1).
    #   ctu_adf_poi_base_knn_kernel.csv  k-NN KERNEL density estimator
    #                                    (baselines.py), F1 64.49% pooled.
    sens = [
        ("ADF vs k-NN density scan (k=25)", "ctu_adf_poi.csv", "ctu_knn_poi_25.csv"),
        ("ADF vs k-NN kernel estimator", "ctu_adf_poi.csv", "ctu_adf_poi_base_knn_kernel.csv"),
        ("ADF vs VBKDE", "ctu_adf_poi.csv", "ctu_adf_poi_base_vbkde.csv"),
        ("ADF vs NW", "ctu_adf_poi.csv", "ctu_adf_poi_base_nw_tiesafe.csv"),
    ]
    for label, fa, fb in sens:
        a, b = cache[fa], cache[fb]
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        print(f"\n  {label}")
        report("all flights, F1", (a["F1"] - b["F1"]).to_numpy(), rng)
        keep = a.index[(a["n_sel"] >= MIN_SEL) & (b["n_sel"] >= MIN_SEL)]
        n_drop = len(a) - len(keep)
        report(f"n_sel >= {MIN_SEL}, F1 ({n_drop} dropped)",
               (a.loc[keep, "F1"] - b.loc[keep, "F1"]).to_numpy(), rng)

    ns = cache["ctu_adf_poi.csv"]["n_sel"]
    print(f"\n  selected-points-per-flight distribution: min={int(ns.min())} "
          f"p10={int(np.percentile(ns,10))} median={int(ns.median())} "
          f"max={int(ns.max())}")
    print(f"  flights with n_sel < {MIN_SEL}: {int((ns < MIN_SEL).sum())} of {len(ns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
