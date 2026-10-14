"""Derive the dataset-description statistics from the repository files.

Everything here is computed from files in ``data/``; nothing is taken on trust
and nothing is invented. The result feeds the dataset paragraph, which must
stay consistent with the provider's restriction that the raw trajectory data
cannot be released.

Usage:
    python diagnostics/dataset_stats.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
DATA = CASE / "data"

EVAL_GEOJSON = DATA / "2024-12-16-CTU_processed.geojson"
REFERENCE = DATA / "poi_without_ctu.csv"
BASELINE = DATA / "2024-12-16-CTU_processed_poi.csv"


def bbox(lon, lat):
    return (float(np.min(lon)), float(np.min(lat)),
            float(np.max(lon)), float(np.max(lat)))


def main():
    print("=== evaluation set (trajectories) ===")
    with open(EVAL_GEOJSON) as fh:
        gj = json.load(fh)
    feats = gj["features"]
    flights, dts, lons, lats = {}, [], [], []
    for f in feats:
        p = f["properties"]
        lon, lat, _alt = f["geometry"]["coordinates"]
        flights.setdefault(p["flight_id"], 0)
        flights[p["flight_id"]] += 1
        lons.append(lon)
        lats.append(lat)
        if "dt" in p:
            try:
                dts.append(float(p["dt"]))
            except (TypeError, ValueError):
                pass

    counts = np.array(list(flights.values()))
    print(f"  flights                : {len(flights):,}")
    print(f"  trajectory points      : {len(feats):,}")
    print(f"  points per flight      : min {counts.min():,}  "
          f"median {int(np.median(counts)):,}  max {counts.max():,}")
    if dts:
        dt = np.asarray(dts, dtype=float)
        dt = dt[np.isfinite(dt)]
        print(f"  sampling interval (s)  : median {np.median(dt):.1f}  "
              f"IQR [{np.percentile(dt, 25):.1f}, {np.percentile(dt, 75):.1f}]  "
              f"n={len(dt):,}")
    mn_lon, mn_lat, mx_lon, mx_lat = bbox(lons, lats)
    print(f"  lon range              : {mn_lon:.4f} .. {mx_lon:.4f}")
    print(f"  lat range              : {mn_lat:.4f} .. {mx_lat:.4f}")

    print("\n=== reference POI set ===")
    ref = pd.read_csv(REFERENCE)
    b = bbox(ref["lon"], ref["lat"])
    s = ref["poi_score"]
    print(f"  POIs                   : {len(ref):,}")
    print(f"  lon range              : {b[0]:.4f} .. {b[2]:.4f}")
    print(f"  lat range              : {b[1]:.4f} .. {b[3]:.4f}")
    print(f"  score range            : {s.min():.4g} .. {s.max():.4g}  "
          f"(median {s.median():.4g})")

    print("\n=== kinematic reference labels used for comparison ===")
    base = pd.read_csv(BASELINE)
    print(f"  baseline POIs          : {len(base):,}")
    print(f"  flights with baseline  : {base['flight_id'].nunique():,}")

    print("\n=== derived split ===")
    print("  reference set : all POIs EXCLUDING the evaluation region (CTU)")
    print("  evaluation set: flights over CTU on 2024-12-16, not in the reference set")


if __name__ == "__main__":
    main()
