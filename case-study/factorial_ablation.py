"""2x2 factorial ablation of the ADF kernel, plus a calibrated mapping sweep.

Refactor
--------
The kernel is parameterised by three switches:

    use_score_weight : amplitude weight  s (True)  or  1 (False)
    use_adaptive_bw  : bandwidth sigma0/(s + eps) (True)  or  sigma0 (False)
    mapping          : score -> bandwidth shape, used when use_adaptive_bw

Factorial cells
---------------
    fact_neither        weight=1  fixed sigma     plain k-NN kernel density
    fact_score_only     weight=s  fixed sigma     reproduces ctu_adf_poi_fixed_500.csv
    fact_adaptive_only  weight=1  sigma0/(s+eps)  the previously missing cell
    fact_both           weight=s  sigma0/(s+eps)  reproduces ctu_adf_poi.csv

Why this is a controlled comparison
-----------------------------------
FAISS retrieval depends only on the query point and k, not on the kernel
parameters. The search is therefore performed ONCE per trajectory point and
every cell is evaluated on the SAME neighbour set, with the same label set,
the same per-trajectory threshold rule and the same matching tolerance. The
cells differ only in the kernel.

Mapping sweep
-------------
All mappings are calibrated to a common MEDIAN bandwidth over the observed
score distribution, so the comparison probes the mapping *shape* rather than
an incidental overall scale shift. The resulting bandwidth range is reported
for each mapping so any residual scale difference is visible rather than
confounded. ``exp_wide`` deliberately uses a much wider range and is a probe
beyond the three mappings named in the plan.

Usage
-----
    python factorial_ablation.py           # compute, write CSVs, print parity
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

DATA = HERE / "data"
SIGMA0 = 500.0
K = 100
EPS = 1e-6
PERCENT = 75
NPROBE = 16
NLIST = 4096

AXIS = 6378137.0
FLATTENING = 1 / 298.257223563
ECC2 = FLATTENING * (2 - FLATTENING)


def geodetic2ecef(lon, lat, hei):
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    n = AXIS / np.sqrt(1 - ECC2 * np.sin(lat) ** 2)
    return np.array([
        (n + hei) * np.cos(lat) * np.cos(lon),
        (n + hei) * np.cos(lat) * np.sin(lon),
        (n * (1 - ECC2) + hei) * np.sin(lat),
    ])


# ---------------------------------------------------------------- mappings --
def build_mappings(scores: np.ndarray, sigma0: float = SIGMA0) -> dict:
    """Calibrate every mapping to a common median bandwidth at this ``sigma0``."""
    s_min = float(scores.min())
    s_med = float(np.median(scores))
    s_max = float(scores.max())
    target_med = sigma0 / s_med
    inv_ratio = (sigma0 / s_min) / (sigma0 / s_max)      # == s_max / s_min

    alpha_narrow = np.log(inv_ratio) / (s_max - s_min)
    b_narrow = target_med / np.exp(-alpha_narrow * s_med)
    alpha_wide = np.log(2.0) / (s_max - s_min)
    b_wide = target_med / np.exp(-alpha_wide * s_med)

    def inv(s):
        return sigma0 / (s + EPS)

    def invshift(s):
        return (target_med * (1.0 + s_med)) / (1.0 + s)

    def exp_narrow(s):
        return b_narrow * np.exp(-alpha_narrow * s)

    def exp_wide(s):
        return b_wide * np.exp(-alpha_wide * s)

    return {
        "inv": inv,
        "invshift": invshift,
        "exp": exp_narrow,
        "exp_wide": exp_wide,
    }


# ------------------------------------------------------------------ kernel --
def field_value(x, neighbors, scores, *, use_score_weight, use_adaptive_bw, sigma_fn,
                sigma0=SIGMA0):
    """One ADF field value. Mirrors the original arithmetic exactly."""
    diff = neighbors - x
    weights = scores if use_score_weight else np.ones_like(scores)

    if use_adaptive_bw:
        sigma = sigma_fn(scores)
        inverse = 1.0 / (sigma ** 2)
        quadratic = np.sum(diff ** 2 * inverse[:, None], axis=1)
    else:
        inverse = 1.0 / (sigma0 ** 2)
        quadratic = np.sum(diff ** 2 * inverse, axis=1)

    return float(np.sum(weights * np.exp(-0.5 * quadratic)))


def main() -> int:
    ref = pd.read_csv(DATA / "poi_without_ctu.csv")
    pos = np.vstack([
        geodetic2ecef(lon, lat, alt)
        for lon, lat, alt in ref[["lon", "lat", "alt"]].to_numpy()
    ]).astype("float32")
    s_all = ref["poi_score"].to_numpy()

    configs = {
        "fact_neither":       dict(use_score_weight=False, use_adaptive_bw=False, mapping="inv", sigma0=SIGMA0),
        "fact_score_only":    dict(use_score_weight=True,  use_adaptive_bw=False, mapping="inv", sigma0=SIGMA0),
        "fact_adaptive_only": dict(use_score_weight=False, use_adaptive_bw=True,  mapping="inv", sigma0=SIGMA0),
        "fact_both":          dict(use_score_weight=True,  use_adaptive_bw=True,  mapping="inv", sigma0=SIGMA0),
        "fact_map_invshift":  dict(use_score_weight=True,  use_adaptive_bw=True,  mapping="invshift", sigma0=SIGMA0),
        "fact_map_exp":       dict(use_score_weight=True,  use_adaptive_bw=True,  mapping="exp", sigma0=SIGMA0),
        "fact_map_exp_wide":  dict(use_score_weight=True,  use_adaptive_bw=True,  mapping="exp_wide", sigma0=SIGMA0),
    }

    # --- sigma0 sweep (R1.4) -------------------------------------------------
    # sigma0 is a KERNEL parameter, not an index parameter: the retrieval below
    # is shared across every configuration, so each extra sigma0 costs kernel
    # evaluation only -- no index rebuild and no extra index.search.
    #
    # Two families exist because the repository previously varied sigma0 only in
    # the DECOUPLED form (bandwidth.py sets `sigma = sigma0`, which deletes the
    # score->bandwidth term).  No prior variant therefore answers R1.4 for the
    # ADF operator as defined (Eq. 3: sigma_j = sigma0 / (s_j + eps)).
    #   sigma0_<v>   coupled    sigma_j = v / (s_j + eps)   <- the ADF operator
    #   fixedbw_<v>  decoupled  sigma_j = v                 <- what bandwidth.py ran
    for _v in (100.0, 250.0, 1000.0, 2000.0):
        configs[f"sigma0_{_v:.0f}"] = dict(
            use_score_weight=True, use_adaptive_bw=True, mapping="inv", sigma0=_v)
    for _v in (100.0, 250.0, 500.0, 1000.0, 2000.0):
        configs[f"fixedbw_{_v:.0f}"] = dict(
            use_score_weight=True, use_adaptive_bw=False, mapping="inv", sigma0=_v)

    sigma0_values = sorted({cfg["sigma0"] for cfg in configs.values()})
    mappings_by_sigma0 = {s0: build_mappings(s_all, s0) for s0 in sigma0_values}
    s_at = np.array([s_all.min(), float(np.median(s_all)), s_all.max()])

    print("=== sigma0 scale reference (coupled 'inv' mapping) ===")
    print(f"  scores: min={s_all.min():.6g}  median={np.median(s_all):.6g}  max={s_all.max():.6g}")
    for s0 in sigma0_values:
        bws = np.asarray(mappings_by_sigma0[s0]["inv"](s_at), dtype=float)
        print(f"  sigma0={s0:7.0f}  ->  sigma {bws[0]:7.1f} @s_min  {bws[1]:7.1f} @median  "
              f"{bws[2]:7.1f} @s_max   range ratio {bws.max() / bws.min():.3f}")

    print("\n=== mapping shapes at sigma0=500, calibrated to common median bw ===")
    for name, fn in mappings_by_sigma0[SIGMA0].items():
        vals = np.asarray(fn(s_at), dtype=float)
        print(f"  {name:<9} sigma {vals[0]:7.1f} @s_min  {vals[1]:7.1f} @median  "
              f"{vals[2]:7.1f} @s_max   range ratio {vals.max() / vals.min():.3f}")

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

        # one retrieval per point, shared by every configuration
        _, idx = index.search(track_ecef, K)
        values = {name: np.empty(len(coords)) for name in configs}

        for j, x in enumerate(track_ecef):
            nb = pos[idx[j]]
            sc = s_all[idx[j]]
            for name, cfg in configs.items():
                values[name][j] = field_value(
                    x, nb, sc,
                    use_score_weight=cfg["use_score_weight"],
                    use_adaptive_bw=cfg["use_adaptive_bw"],
                    sigma_fn=mappings_by_sigma0[cfg["sigma0"]][cfg["mapping"]],
                    sigma0=cfg["sigma0"],
                )

        n_points += len(coords)
        for name, vals in values.items():
            thr = np.percentile(vals, PERCENT)
            mask = vals >= thr
            for (lon, lat, alt), v, m in zip(coords, vals, mask):
                rows[name].append([fid, lon, lat, alt, v, int(m)])

    elapsed = time.perf_counter() - t0
    print(f"\nshared-retrieval pass: {n_points:,} points, {elapsed:.1f} s "
          f"({len(configs)} configurations)")

    for name in configs:
        out = pd.DataFrame(rows[name], columns=["flight_id", "lon", "lat", "alt", "ADF", "POI"])
        path = DATA / f"ctu_adf_poi_{name}.csv"
        out.to_csv(path, index=False)
        print(f"  wrote {path.name:<34} n_poi={int(out['POI'].sum()):,}")

    # ------------------------------------------------------------- parity --
    print("\n=== parity with existing canonical outputs ===")
    for new, old in (("fact_both", "ctu_adf_poi.csv"),
                     ("fact_score_only", "ctu_adf_poi_fixed_500.csv"),
                     ("fixedbw_500", "ctu_adf_poi_fixed_500.csv")):
        a = pd.read_csv(DATA / f"ctu_adf_poi_{new}.csv")
        b = pd.read_csv(DATA / old)
        if len(a) != len(b):
            print(f"  {new:<18} vs {old:<32} ROW COUNT DIFFERS")
            continue
        key_a = list(zip(a["flight_id"], a["lon"], a["lat"], a["alt"]))
        key_b = list(zip(b["flight_id"], b["lon"], b["lat"], b["alt"]))
        same_geom = key_a == key_b
        if not same_geom:
            print(f"  {new:<18} vs {old:<32} ROW ORDER/GEOMETRY DIFFERS")
            continue
        same_mask = bool((a["POI"].to_numpy() == b["POI"].to_numpy()).all())
        max_diff = float(np.abs(a["ADF"].to_numpy() - b["ADF"].to_numpy()).max())
        print(f"  {new:<18} vs {old:<32} POI identical={same_mask}  "
              f"max |dADF|={max_diff:.3e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
