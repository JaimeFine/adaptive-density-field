"""Score-normalization sensitivity + epsilon-floor sensitivity (plan item R1.3).

Why three of the five requested normalizations cannot be run
-----------------------------------------------------------
The bandwidth is sigma_j = sigma0 / (s_j + eps). A normalization is only
admissible if it keeps s_j strictly positive AND bounded away from zero:

    raw        s in [0.75, 1.00]      -> sigma in [500, 667] m      ADMISSIBLE
    softplus   s in [1.137, 1.313]    -> sigma in [381, 440] m      ADMISSIBLE
    min-max    s in [0, 1]            -> s = 0 at the minimum,       ILL-POSED
                                         sigma -> infinity
    z-score    s ~ N(0, 1)            -> s <= 0 for ~half the points, ILL-POSED
                                         sigma negative or infinite
    rank       s in (0, 1]            -> sigma_max ~ 1e9 m           ILL-POSED
                                         (no longer a spatial scale)

This is not a numerical nuisance; it is a property of the formulation. A
normalization that produces non-positive scores cannot be used with an inverse
score-to-bandwidth mapping at all, so the "sensitivity" of the operator to
normalization is not a free robustness check: the choice of normalization
changes the set of admissible formulations.

Decisions taken (user unavailable; review before use)
----------------------------------------------------
1. The three ill-posed normalizations are reported analytically and NOT run
   numerically. The two admissible ones plus two strictly-positive rescalings
   are run:
       raw        baseline
       softplus   pointwise, positive
       minmax_1_2 min-max onto [1, 2]     (the admissible variant of min-max)
       rank_1_2   uniform rank onto [1, 2] (the admissible variant of rank)
2. Each normalization is evaluated as-is AND re-calibrated to a common median
   bandwidth, so the score-distribution *shape* is separated from the
   bandwidth *range*.
3. Normalization is applied to both channels (what a practitioner would
   naively do), plus a bandwidth-only variant that holds the amplitude weight
   at the RAW score.
4. Epsilon floor swept over {1e-6, 1e-4, 1e-2, 1e-1, 2.5e-1}. With s_min=0.75
   the floor is negligible until eps ~ 1e-1.

Retrieval is parameter-independent, so the FAISS search is performed ONCE per
trajectory point and every configuration is evaluated on the same neighbour
set, as in factorial_ablation.py.

Usage
-----
    python normalization_sensitivity.py
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import faiss  # noqa: E402
import pandas as pd  # noqa: E402

from factorial_ablation import (  # noqa: E402
    DATA, SIGMA0, K, PERCENT, NPROBE, NLIST,
    field_value, geodetic2ecef,
)

TARGET_MEDIAN = 601.7      # m, the median bandwidth of the raw formulation
DEFAULT_EPS = 1e-6

# normalizations whose bandwidth may become non-positive or unbounded
ILL_POSED = ["min-max to [0,1]", "z-score", "rank to (0,1]"]

def build_normalizers(s_all: np.ndarray) -> dict:
    """Each normalizer maps an arbitrary score array to positive scores."""
    s_min = float(s_all.min())
    s_max = float(s_all.max())
    span = s_max - s_min

    order = np.argsort(s_all)
    sorted_s = s_all[order]
    ranks = np.arange(1, len(s_all) + 1, dtype=float)

    def raw(s):
        return np.asarray(s, dtype=float)

    def softplus(s):
        return np.log1p(np.exp(np.asarray(s, dtype=float)))

    def minmax_1_2(s):
        return 1.0 + (np.asarray(s, dtype=float) - s_min) / span

    def rank_1_2(s):
        # rank as a function of score value, interpolated for arbitrary inputs
        r = np.interp(np.asarray(s, dtype=float), sorted_s, ranks)
        return 1.0 + (r - 1.0) / (len(s_all) - 1.0)

    return {
        "raw": raw,
        "softplus": softplus,
        "minmax_1_2": minmax_1_2,
        "rank_1_2": rank_1_2,
    }

def _sigma_from_normalized(eps: float, s0: float):
    """sigma = s0 / (s' + eps) for scores that are already normalized."""
    return lambda s: s0 / (s + eps)

def _sigma_from_normalizer(eps: float, fn):
    """sigma = sigma0 / (norm(s) + eps) for raw scores."""
    return lambda s: SIGMA0 / (fn(s) + eps)

def main() -> int:
    ref = pd.read_csv(DATA / "poi_without_ctu.csv")
    pos = np.vstack([
        geodetic2ecef(lon, lat, alt)
        for lon, lat, alt in ref[["lon", "lat", "alt"]].to_numpy()
    ]).astype("float32")
    s_all = ref["poi_score"].to_numpy(dtype=float)

    print("=== admissibility of the requested normalizations ===")
    print("  admissible : raw, softplus")
    print("  rescaled   : min-max -> [1,2], rank -> [1,2]  (strictly positive)")
    print(f"  NOT RUN    : {', '.join(ILL_POSED)}")
    print("               sigma_j = sigma0/(s_j+eps) is undefined for s_j <= 0")
    print("               and unbounded as s_j -> 0.")

    norms = build_normalizers(s_all)
    # normalizing the full reference array once lets the loop index it in O(1)
    norm_scores = {name: fn(s_all) for name, fn in norms.items()}

    print("\n=== normalization -> bandwidth (sigma0 = 500 m) ===")
    sigma0_eff = {}
    for name, fn in norms.items():
        lo = float(fn(np.array([s_all.min()]))[0])
        hi = float(fn(np.array([s_all.max()]))[0])
        med = float(fn(np.array([np.median(s_all)]))[0])
        s_lo, s_hi = SIGMA0 / lo, SIGMA0 / hi
        sigma0_eff[name] = TARGET_MEDIAN * med
        print(f"  {name:<11} s' in [{lo:.3f}, {hi:.3f}]  "
              f"sigma in [{s_hi:6.1f}, {s_lo:6.1f}] m  range x{s_lo / s_hi:.3f}"
              f"   calibrated sigma0={sigma0_eff[name]:.1f}")

    configs: dict = {}
    for name in ("raw", "softplus", "minmax_1_2", "rank_1_2"):
        configs[f"norm_{name}"] = dict(norm=name, channel="both", cal=False, eps=DEFAULT_EPS)
    for name in ("softplus", "minmax_1_2", "rank_1_2"):
        configs[f"norm_{name}_bw"] = dict(norm=name, channel="bw", cal=False, eps=DEFAULT_EPS)
        configs[f"norm_{name}_cal"] = dict(norm=name, channel="both", cal=True, eps=DEFAULT_EPS)
    for eps in (1e-4, 1e-2, 1e-1, 2.5e-1):
        configs[f"eps_{eps:g}"] = dict(norm="raw", channel="both", cal=False, eps=eps)

    quantizer = faiss.IndexFlatL2(3)
    index = faiss.IndexIVFFlat(quantizer, 3, NLIST)
    index.train(pos)
    index.add(pos)
    index.nprobe = NPROBE

    with open(DATA / "2024-12-16-CTU_processed.geojson") as fh:
        track = json.load(fh)
    flights: dict = {}
    for feature in track["features"]:
        fid = feature["properties"]["flight_id"]
        flights.setdefault(fid, []).append(feature["geometry"]["coordinates"])

    rows = {name: [] for name in configs}
    t0 = time.perf_counter()
    n_points = 0

    for fid, coords in flights.items():
        coords = np.array(coords)
        track_ecef = np.vstack([
            geodetic2ecef(lon, lat, alt) for lon, lat, alt in coords
        ]).astype("float32")
        _, idx = index.search(track_ecef, K)
        values = {name: np.empty(len(coords)) for name in configs}

        for j, x in enumerate(track_ecef):
            nb = pos[idx[j]]
            sc_raw = s_all[idx[j]]

            for name, cfg in configs.items():
                nname = cfg["norm"]
                if cfg["channel"] == "both":
                    scores_arg = norm_scores[nname][idx[j]]
                    s0 = sigma0_eff[nname] if cfg["cal"] else SIGMA0
                    sigma_fn = _sigma_from_normalized(cfg["eps"], s0)
                else:
                    scores_arg = sc_raw
                    sigma_fn = _sigma_from_normalizer(cfg["eps"], norms[nname])

                values[name][j] = field_value(
                    x, nb, scores_arg,
                    use_score_weight=True, use_adaptive_bw=True,
                    sigma_fn=sigma_fn,
                )

        n_points += len(coords)
        for name, vals in values.items():
            thr = np.percentile(vals, PERCENT)
            mask = vals >= thr
            for (lon, lat, alt), v, m in zip(coords, vals, mask):
                rows[name].append([fid, lon, lat, alt, v, int(m)])

    print(f"\nshared-retrieval pass: {n_points:,} points, "
          f"{time.perf_counter() - t0:.1f} s ({len(configs)} configurations)")

    for name in configs:
        out = pd.DataFrame(rows[name], columns=["flight_id", "lon", "lat", "alt", "ADF", "POI"])
        path = DATA / f"ctu_adf_poi_{name}.csv"
        out.to_csv(path, index=False)
        print(f"  wrote {path.name:<38} n_poi={int(out['POI'].sum()):,}")

    print("\n=== parity: norm_raw must reproduce the canonical adaptive output ===")
    a = pd.read_csv(DATA / "ctu_adf_poi_norm_raw.csv")
    b = pd.read_csv(DATA / "ctu_adf_poi_fact_both.csv")
    key_a = list(zip(a["flight_id"], a["lon"], a["lat"], a["alt"]))
    key_b = list(zip(b["flight_id"], b["lon"], b["lat"], b["alt"]))
    if key_a != key_b:
        print("  ROW ORDER/GEOMETRY DIFFERS")
    else:
        same = bool((a["POI"].to_numpy() == b["POI"].to_numpy()).all())
        md = float(np.abs(a["ADF"].to_numpy() - b["ADF"].to_numpy()).max())
        print(f"  POI identical={same}  max |dADF|={md:.3e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
