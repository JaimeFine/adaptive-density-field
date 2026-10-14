"""Baseline suite for R2.1 / R2.4: comparable comparators on one protocol.

Protocol (identical for every baseline and for ADF)
---------------------------------------------------
Same reference set (2,243,080 POIs), same 268 evaluation flights, same
per-trajectory P75 selection rule, same output schema (ADF, POI) so the same
metric harness consumes them, and the same 3D WGS84 ECEF matching tolerance.

Which factor each baseline varies
---------------------------------
The point of a baseline suite is that each comparator isolates one design
choice. This suite states that explicitly:

    base_kde_tree      exact fixed-bandwidth KDE (sigma=500 m), unit amplitude
                       -> the tree/ANN-accelerated KDE baseline. Isolates the
                          kernel-evaluation strategy (exact tree vs the IVF
                          k-NN truncation ADF uses).
    base_vbkde         Abramson variable-bandwidth KDE: bandwidth driven by the
                       estimated local DENSITY at each reference point,
                       amplitude 1/h^d (density-consistent)
                       -> isolates the BANDWIDTH DRIVER: density (endogenous)
                          versus ADF's external score (exogenous).
    base_vbkde_amp1    same density-driven bandwidths, unit amplitude
                       -> separates the bandwidth rule from the 1/h^d
                          normaliser.
    base_knn_kernel    classic k-NN density estimator k / (n V r_k^d)
                       -> isolates the estimator family (nearest-neighbour
                          distance vs kernel summation).
    base_nw            Nadaraya-Watson / kernel attention with the SAME adaptive
                       sigma_j and the SAME neighbour set as ADF, but normalised
                       by the local kernel mass
                       -> isolates the absence of normalisation, which is the
                          defining choice of ADF's field.

Already covered elsewhere (NOT recomputed here)
-----------------------------------------------
    ladder (fixed KDE -> adaptive KDE -> weighted adaptive kernel -> ADF)
        is exactly the 2x2 factorial: fact_neither -> fact_adaptive_only ->
        fact_score_only -> fact_both.
    kernel-attention is algebraically identical to Nadaraya-Watson: the softmax
        of -||x-x_j||^2/(2 sigma^2) equals K_j / sum_k K_k, so the
        attention-weighted value sum_j a_j s_j is the NW local mean.

Implementation note on the tree baselines
-----------------------------------------
The Gaussian has negligible mass beyond 4 sigma, so querying a cKDTree within
a radius of 4 x (largest bandwidth) and summing analytically reproduces the
exact KDE to numerical precision. This truncation is NUMERICAL; ADF's k-NN
truncation is STRUCTURAL. That distinction is the point of base_kde_tree.

Usage
-----
    python baselines.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import faiss  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from factorial_ablation import (  # noqa: E402
    DATA, SIGMA0, K, PERCENT, NPROBE, NLIST, geodetic2ecef,
)

RHO_K = 20          # neighbourhood size for the local-density proxy
CHUNK = 200_000     # chunk size for the reference k-NN distance computation
CLIP_LO, CLIP_HI = 1.0, 99.0   # percentile clip on implied bandwidths


def main() -> int:
    ref = pd.read_csv(DATA / "poi_without_ctu.csv")
    pos32 = np.vstack([
        geodetic2ecef(lon, lat, alt)
        for lon, lat, alt in ref[["lon", "lat", "alt"]].to_numpy()
    ]).astype("float32")
    pos = pos32.astype("float64")
    s_all = ref["poi_score"].to_numpy(dtype=float)
    n = len(ref)
    print(f"reference set: n = {n:,}")

    # ---------------------------------------------------------- local density
    t0 = time.perf_counter()
    tree = cKDTree(pos)
    d_k = np.empty(n, dtype="float64")
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        # k = RHO_K + 1 because the query point is its own nearest neighbour
        dist, _ = tree.query(pos[start:stop], k=RHO_K + 1, workers=-1)
        d_k[start:stop] = dist[:, RHO_K]
    print(f"reference {RHO_K}-NN distances: {time.perf_counter() - t0:.1f} s")

    # Abramson: h ~ f^(-1/(d+4)); with f ~ d_k^-3 in 3D this gives h ~ d_k^(3/7)
    d_med = float(np.median(d_k))
    h_j = SIGMA0 * (d_k / d_med) ** (3.0 / 7.0)
    h_lo, h_hi = np.percentile(h_j, [CLIP_LO, CLIP_HI])
    n_clipped = int(((h_j < h_lo) | (h_j > h_hi)).sum())
    h_j = np.clip(h_j, h_lo, h_hi)
    inv_h3 = 1.0 / h_j ** 3
    print(f"VBKDE bandwidth h: p{CLIP_LO:g}={h_lo:.1f} m  median={np.median(h_j):.1f} m  "
          f"p{CLIP_HI:g}={h_hi:.1f} m  (clipped {n_clipped:,} points)")
    print(f"  range ratio h_hi/h_lo = {h_hi / h_lo:.2f}  "
          f"(ADF's score-driven range ratio is 1.33)")

    R_KDE = 4.0 * SIGMA0
    R_VB = 4.0 * float(h_hi)
    R = max(R_KDE, R_VB)
    print(f"tree query radius: {R:.0f} m (4 x max bandwidth)")

    # ------------------------------------------------------------- retrieval
    quantizer = faiss.IndexFlatL2(3)
    index = faiss.IndexIVFFlat(quantizer, 3, NLIST)
    index.train(pos32)
    index.add(pos32)
    index.nprobe = NPROBE

    with open(DATA / "2024-12-16-CTU_processed.geojson") as fh:
        track = json.load(fh)
    flights: dict = {}
    for feature in track["features"]:
        fid = feature["properties"]["flight_id"]
        flights.setdefault(fid, []).append(feature["geometry"]["coordinates"])

    names = ["base_kde_tree", "base_vbkde", "base_vbkde_amp1",
             "base_knn_kernel", "base_nw"]
    rows = {nm: [] for nm in names}
    t0 = time.perf_counter()

    for fid, coords in flights.items():
        coords = np.array(coords)
        track_ecef = np.vstack([
            geodetic2ecef(lon, lat, alt) for lon, lat, alt in coords
        ]).astype("float32")
        # ADF-consistent neighbour set for the NW baseline.
        # FAISS returns SQUARED L2 distances.
        d_faiss, idx = index.search(track_ecef, K)

        vals = {nm: np.empty(len(coords)) for nm in names}

        for j, x in enumerate(track_ecef):
            # exact tree evaluation of both fixed and variable bandwidth KDEs
            nb = tree.query_ball_point(x.astype("float64"), R)
            if nb:
                nb = np.asarray(nb)
                d2 = np.sum((pos[nb] - x) ** 2, axis=1)
                vals["base_kde_tree"][j] = float(
                    np.sum(np.exp(-0.5 * d2 / SIGMA0 ** 2))
                )
                w = np.exp(-0.5 * d2 / h_j[nb] ** 2)
                vals["base_vbkde"][j] = float(np.sum(inv_h3[nb] * w))
                vals["base_vbkde_amp1"][j] = float(np.sum(w))
            else:
                for nm in ("base_kde_tree", "base_vbkde", "base_vbkde_amp1"):
                    vals[nm][j] = 0.0

            # k-NN density estimator: k / (n * V_d * r_k^d)
            r_k = float(np.sqrt(max(d_faiss[j, K - 1], 0.0)))
            vol = (4.0 / 3.0) * np.pi * max(r_k, 1e-9) ** 3
            vals["base_knn_kernel"][j] = K / (n * vol)

            # Nadaraya-Watson / kernel attention: same sigma_j, normalised
            nb_idx = idx[j]
            sc = s_all[nb_idx]
            dn = pos32[nb_idx] - x
            sig = SIGMA0 / (sc + 1e-6)
            ker = np.exp(-0.5 * np.sum(dn ** 2, axis=1) / sig ** 2)
            tot = float(np.sum(ker))
            vals["base_nw"][j] = float(np.sum(sc * ker) / tot) if tot > 0 else 0.0

        for nm, v in vals.items():
            thr = np.percentile(v, PERCENT)
            mask = v >= thr
            for (lon, lat, alt), vv, m in zip(coords, v, mask):
                rows[nm].append([fid, lon, lat, alt, vv, int(m)])

    print(f"\nevaluation pass: {sum(len(c) for c in flights.values()):,} points, "
          f"{time.perf_counter() - t0:.1f} s")

    for nm in names:
        out = pd.DataFrame(rows[nm], columns=["flight_id", "lon", "lat", "alt", "ADF", "POI"])
        path = DATA / f"ctu_adf_poi_{nm}.csv"
        out.to_csv(path, index=False)
        print(f"  wrote {path.name:<34} n_poi={int(out['POI'].sum()):,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
