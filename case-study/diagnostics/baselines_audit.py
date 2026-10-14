"""Audit of the baseline suite: formulas, confounds, paired tests, latency.

Checks
------
A. Retrieval determinism. Does rebuilding the IVF index reproduce identical
   neighbour lists? This is required before claiming that ``base_nw`` and ADF
   differ only by aggregate normalisation.
B. VBKDE bandwidth rule: recompute the Abramson parameterisation and report
   exact parameters (median, percentiles, range, clipping).
C. Paired per-flight comparisons on the same unit, resampling scheme and
   statistics as ``diagnostics/perflight_stats.py`` (one observation per
   flight, 10,000 flight resamples, seed 0, percentile CI, Cohen's d_z,
   two-sided Wilcoxon signed-rank).
D. Latency: repeated timed runs of the per-point evaluation, with the timing
   protocol and the software/hardware environment recorded.

This script is read-only with respect to results: it writes no CSV outputs and
does not modify the manuscript.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
for p in (str(CASE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import faiss  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from factorial_ablation import (  # noqa: E402
    DATA, SIGMA0, K, PERCENT, NPROBE, NLIST, geodetic2ecef,
)
from metric_compare import geodetic_to_ecef  # noqa: E402

RHO_K = 20
CHUNK = 200_000
CLIP_LO, CLIP_HI = 1.0, 99.0
EXPONENT = 3.0 / 7.0
N_BOOT = 10_000
SEED = 0
T = 200.0
N_REP = 5
BASELINE = DATA / "2024-12-16-CTU_processed_poi.csv"
EPS = 1e-6


# --------------------------------------------------------------- utilities --
def load_flights() -> dict:
    with open(DATA / "2024-12-16-CTU_processed.geojson") as fh:
        track = json.load(fh)
    flights: dict = {}
    for f in track["features"]:
        flights.setdefault(f["properties"]["flight_id"], []).append(
            f["geometry"]["coordinates"]
        )
    return flights


def build_index(pos32: np.ndarray):
    q = faiss.IndexFlatL2(3)
    ix = faiss.IndexIVFFlat(q, 3, NLIST)
    ix.train(pos32)
    ix.add(pos32)
    ix.nprobe = NPROBE
    return ix


def load_selected(path: Path) -> pd.DataFrame:
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
        s_tree = cKDTree(s_xyz)
        b_tree = cKDTree(b)
        tt = int((b_tree.query(s_xyz, k=1)[0] <= threshold).sum())
        n_sel = len(s_xyz)
        fn = int((s_tree.query(b, k=1)[0] > threshold).sum())
        p = tt / n_sel if n_sel else np.nan
        r = tt / (tt + fn) if (tt + fn) else np.nan
        f1 = 2 * p * r / (p + r) if (p + r) else np.nan
        rows.append(dict(flight_id=fid, Precision=p, Recall=r, F1=f1))
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
        pval = float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        pval = float("nan")
    print(f"  {name:<16} mean D={mean:+.4f}  sd={sd:.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  d_z={dz:+.3f}  p={pval:.3g}")
    return mean, lo, hi, dz, pval


def main() -> int:
    t_start = time.perf_counter()

    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    print(f"  python  : {sys.version.split()[0]}  ({platform.machine()})")
    print(f"  numpy   : {np.__version__}   pandas: {pd.__version__}   "
          f"faiss: {faiss.__version__}")
    print(f"  platform: {platform.platform()}")
    print(f"  cpu     : {platform.processor()!r}  logical cores: {os.cpu_count()}")
    print("  note    : no GPU used; all timing below is single-process CPU.")

    ref = pd.read_csv(DATA / "poi_without_ctu.csv")
    pos64 = np.vstack([
        geodetic2ecef(lon, lat, alt)
        for lon, lat, alt in ref[["lon", "lat", "alt"]].to_numpy()
    ]).astype("float64")
    pos32 = pos64.astype("float32")
    s_all = ref["poi_score"].to_numpy(dtype=float)
    n_ref = len(ref)

    # ---------------------------------------------------------------- check A
    print("\n" + "=" * 72)
    print("CHECK A - retrieval determinism (does rebuilding the IVF index")
    print("          reproduce the same neighbour lists?)")
    print("=" * 72)
    flights = load_flights()
    all_pts = []
    for coords in flights.values():
        for lon, lat, alt in coords:
            all_pts.append(geodetic2ecef(lon, lat, alt))
    all_pts = np.asarray(all_pts, dtype="float32")
    rng = np.random.default_rng(SEED)
    probe = all_pts[rng.choice(len(all_pts), size=3_000, replace=False)]

    t = time.perf_counter()
    ix_a = build_index(pos32)
    t_build = time.perf_counter() - t
    ix_b = build_index(pos32)
    da, ia = ix_a.search(probe, K)
    db, ib = ix_b.search(probe, K)
    print(f"  index train+add time: {t_build:.1f} s each")
    print(f"  neighbour index lists identical : {np.array_equal(ia, ib)}")
    print(f"  distances bit-identical         : {np.array_equal(da, db)}")
    frac = float((ia == ib).all(axis=1).mean())
    print(f"  fraction of probes with identical 100-NN sets: {frac:.4f}")
    if not np.array_equal(ia, ib):
        print("  !! retrieval is NOT deterministic; the base_nw vs ADF")
        print("     'normalisation-only' claim cannot be made from the stored")
        print("     outputs and must be withdrawn or re-derived in one process.")
    index = ix_a

    # ---------------------------------------------------------------- check B
    print("\n" + "=" * 72)
    print("CHECK B - VBKDE bandwidth rule and parameters")
    print("=" * 72)
    tree = cKDTree(pos64)
    d_k = np.empty(n_ref, dtype="float64")
    for start in range(0, n_ref, CHUNK):
        stop = min(start + CHUNK, n_ref)
        dist, _ = tree.query(pos64[start:stop], k=RHO_K + 1, workers=-1)
        d_k[start:stop] = dist[:, RHO_K]
    d_med = float(np.median(d_k))
    h_raw = SIGMA0 * (d_k / d_med) ** EXPONENT
    h_lo = float(np.percentile(h_raw, CLIP_LO))
    h_hi = float(np.percentile(h_raw, CLIP_HI))
    h = np.clip(h_raw, h_lo, h_hi)
    print(f"  rule      : h_j = sigma0 * (d_j / median(d))^({EXPONENT:.4f}), "
          f"d_j = distance to {RHO_K}-th NN")
    print(f"              (Abramson h ~ f^(-1/(d+4)), f ~ d^-3 in 3D)")
    print(f"  d_k       : median={d_med:.1f} m")
    print(f"  h raw     : min={h_raw.min():.1f}  median={np.median(h_raw):.1f}  "
          f"max={h_raw.max():.1f} m")
    print(f"  clip      : [{h_lo:.1f}, {h_hi:.1f}] m at p{CLIP_LO:g}/p{CLIP_HI:g}  "
          f"({int(((h_raw < h_lo) | (h_raw > h_hi)).sum()):,} points)")
    print(f"  h clipped : min={h.min():.1f}  median={np.median(h):.1f}  "
          f"max={h.max():.1f} m   range ratio={h.max() / h.min():.2f}")
    sig_adf = SIGMA0 / s_all
    print(f"  ADF sigma : min={sig_adf.min():.1f}  median={np.median(sig_adf):.1f}  "
          f"max={sig_adf.max():.1f} m   range ratio="
          f"{sig_adf.max() / sig_adf.min():.2f}")
    print(f"  SCALE MISMATCH: VBKDE median {np.median(h):.0f} m vs ADF median "
          f"{np.median(sig_adf):.0f} m  -> ratio {np.median(h) / np.median(sig_adf):.3f}")
    inv_h3 = 1.0 / h ** 3

    # ---------------------------------------------------------------- check C
    print("\n" + "=" * 72)
    print(f"CHECK C - paired per-flight comparison at T = {T:.0f} m")
    print("          unit = one flight; 10,000 flight resamples; seed 0")
    print("=" * 72)
    base = pd.read_csv(BASELINE)
    base_by_flight = {
        fid: geodetic_to_ecef(g["lon"], g["lat"], g["alt"])
        for fid, g in base.groupby("flight_id")
    }
    adf = per_flight_metrics(load_selected(DATA / "ctu_adf_poi.csv"), base_by_flight, T)
    print(f"  flights: {len(adf)}")
    print(f"  {'metric':<12}{'ADF mean':>10}")
    for m in ("Precision", "Recall", "F1"):
        print(f"  {m:<12}{adf[m].mean():>10.4f}")

    pairs = {
        "VBKDE": "ctu_adf_poi_base_vbkde.csv",
        "VBKDE_amp1": "ctu_adf_poi_base_vbkde_amp1.csv",
        "KDE_tree": "ctu_adf_poi_base_kde_tree.csv",
        "NW_tiesafe": "ctu_adf_poi_base_nw_tiesafe.csv",
        "knn_kernel": "ctu_adf_poi_base_knn_kernel.csv",
    }
    rng_c = np.random.default_rng(SEED)
    for label, fname in pairs.items():
        other = per_flight_metrics(load_selected(DATA / fname), base_by_flight, T)
        common = adf.index.intersection(other.index)
        a, o = adf.loc[common], other.loc[common]
        print(f"\n  ADF minus {label}  (paired flights = {len(common)})")
        for m in ("Precision", "Recall", "F1"):
            report(m, (a[m] - o[m]).to_numpy(), rng_c)

    print("\n  reading: intervals exclude 0 => a difference is resolvable at this")
    print("  sample size. Flights share one reference set and one detector, so")
    print("  intervals are optimistically narrow; treat as descriptive. No")
    print("  equivalence is claimed for intervals that include 0.")

    # ---------------------------------------------------------------- check D
    print("\n" + "=" * 72)
    print("CHECK D - latency, repeated runs")
    print("=" * 72)
    print(f"  protocol: {N_REP} timed repetitions per configuration after one")
    print("            discarded warm-up. Each repetition times the FULL loop")
    print("            over all trajectory points for that configuration,")
    print("            including that configuration's own retrieval.")
    print("            Excluded: CSV parsing, tree construction, reference k-NN")
    print("            precompute, IVF training/adding. Single process, CPU only.")

    kdes = {nm: [] for nm in ("adf", "nw", "knn_kernel", "kde_tree", "vbkde")}

    def run_config(kind: str) -> None:
        for fid, coords in flights.items():
            coords = np.array(coords)
            te = np.vstack([
                geodetic2ecef(lon, lat, alt) for lon, lat, alt in coords
            ]).astype("float32")
            if kind in ("adf", "nw", "knn_kernel"):
                d_f, idx = index.search(te, K)
            for j, x in enumerate(te):
                if kind == "adf":
                    sc = s_all[idx[j]]
                    dn = pos32[idx[j]] - x
                    sig = SIGMA0 / (sc + EPS)
                    ker = np.exp(-0.5 * np.sum(dn ** 2, axis=1) / sig ** 2)
                    _ = float(np.sum(sc * ker))
                elif kind == "nw":
                    sc = s_all[idx[j]]
                    dn = pos32[idx[j]] - x
                    sig = SIGMA0 / (sc + EPS)
                    ker = np.exp(-0.5 * np.sum(dn ** 2, axis=1) / sig ** 2)
                    tot = float(np.sum(ker))
                    _ = float(np.sum(sc * ker) / tot) if tot > 0 else 0.0
                elif kind == "knn_kernel":
                    r_k = float(np.sqrt(max(d_f[j, K - 1], 0.0)))
                    _ = K / (n_ref * (4.0 / 3.0) * np.pi * max(r_k, 1e-9) ** 3)
                else:
                    nb = tree.query_ball_point(x.astype("float64"), 5962.4)
                    if nb:
                        nb = np.asarray(nb)
                        d2 = np.sum((pos64[nb] - x) ** 2, axis=1)
                        if kind == "kde_tree":
                            _ = float(np.sum(np.exp(-0.5 * d2 / SIGMA0 ** 2)))
                        else:
                            _ = float(np.sum(inv_h3[nb] *
                                             np.exp(-0.5 * d2 / h[nb] ** 2)))

    n_pts = sum(len(c) for c in flights.values())
    for kind in kdes:
        run_config(kind)  # warm-up, discarded
    for rep in range(N_REP):
        for kind in kdes:
            t0 = time.perf_counter()
            run_config(kind)
            kdes[kind].append(time.perf_counter() - t0)

    print(f"\n  points per repetition: {n_pts:,}")
    print(f"  {'configuration':<14}{'mean s':>10}{'sd s':>9}{'ms/query':>11}")
    for kind, ts in kdes.items():
        mu = float(np.mean(ts))
        sd = float(np.std(ts, ddof=1)) if len(ts) > 1 else 0.0
        print(f"  {kind:<14}{mu:>10.3f}{sd:>9.3f}{mu / n_pts * 1000:>11.5f}")

    print(f"\ntotal audit wall time: {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
