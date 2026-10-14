"""Diagnostic: is the ADF score a proxy for local data density?

This is the decisive test for distinguishing ADF from variable-bandwidth KDE
(VBKDE / Abramson). In Abramson-style adaptive KDE the bandwidth at a data
point is a function of the *estimated local density*: h_i ~ f(x_i)^(-1/(d+4)).

In ADF the bandwidth is sigma_j = sigma0 / s_j, where s_j is an externally
supplied application score. If s_j happened to be a density proxy, ADF would
reduce to a reparameterisation of VBKDE. This script measures whether that is
true.

It is read-only: it computes no field values and writes no data files.

Usage:
    python diagnostics/score_density_independence.py
"""

from __future__ import annotations

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
K = 10
N_SAMPLE = 200_000
SEED = 0


def main() -> None:
    ref = pd.read_csv(DATA / "poi_without_ctu.csv")
    lon = ref["lon"].to_numpy(dtype=float)
    lat = ref["lat"].to_numpy(dtype=float)
    alt = ref["alt"].to_numpy(dtype=float)
    score = ref["poi_score"].to_numpy(dtype=float)
    n = len(ref)
    print("=== Score vs local density (reference POI set) ===")
    print(f"  n = {n:,}")

    xyz = geodetic_to_ecef(lon, lat, alt).astype("float64")
    tree = cKDTree(xyz)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=min(N_SAMPLE, n), replace=False)
    # k=K+1 because the query point is its own nearest neighbour.
    dist, _ = tree.query(xyz[idx], k=K + 1, workers=-1)
    d_k = dist[:, K]                      # distance to the K-th nearest neighbour
    s = score[idx]

    p10, med, p90 = np.percentile(d_k, [10, 50, 90])
    print("\n--- local neighbourhood scale, sampled points ---")
    print(f"  distance to {K}-th NN: p10={p10:,.1f} m  median={med:,.1f} m  p90={p90:,.1f} m")
    print(f"  sample score range: {s.min():.4g} .. {s.max():.4g} "
          f"(median {np.median(s):.4g})")

    rho, pval = spearmanr(s, d_k)
    rho_log, pval_log = spearmanr(s, np.log(d_k))
    print("\n--- is the score a density proxy? ---")
    print(f"  Spearman(s, d_k)     = {rho:+.4f}   (p = {pval:.3g})")
    print(f"  Spearman(s, log d_k) = {rho_log:+.4f}   (p = {pval_log:.3g})")
    print("  larger d_k means LOWER local density, so a density proxy would")
    print("  show a clearly negative correlation here.")

    # Contrast the bandwidth spread each mechanism would produce.
    print("\n--- bandwidth spread under each mechanism ---")
    adf_lo, adf_hi = SIGMA0 / s.max(), SIGMA0 / s.min()
    print(f"  ADF score-driven sigma: {adf_lo:,.0f} .. {adf_hi:,.0f} m "
          f"(ratio {adf_hi / adf_lo:.3f})")
    # Abramson in d=3: h ~ f^(-1/(d+4)) = f^(-1/7); with f ~ d_k^-3, h ~ d_k^(3/7).
    density_ratio = (p90 / p10) ** (3.0 / 7.0)
    print(f"  density-driven sigma (Abramson, d=3, h ~ f^(-1/7)) would span a")
    print(f"  ratio of about {density_ratio:.2f} across the same p10..p90 range")

    print("\n--- verdict ---")
    if abs(rho) < 0.2:
        print("  |rho| < 0.2: the score carries essentially no local-density")
        print("  information. ADF's bandwidth driver is therefore exogenous and")
        print("  distinct from the density-driven bandwidth of VBKDE.")
    else:
        print("  |rho| >= 0.2: the score co-varies with local density. The")
        print("  distinction from VBKDE needs stronger evidence than this test.")


if __name__ == "__main__":
    main()
