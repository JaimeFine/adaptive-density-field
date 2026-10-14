"""Check whether the global lon/lat extremes are outliers or genuine coverage.

Motivation: ``dataset_stats.py`` reported lon spans of ~360 deg for both the
evaluation set and the reference set, which contradicts the manuscript's
"nationwide" / "Chengdu region" description. Before writing any dataset
paragraph we need to know whether those extremes are a handful of corrupted
coordinates or real coverage.

Usage:
    python diagnostics/coverage_check.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# Chengdu (CTU) neighbourhood box
CTU = dict(lon=(102.0, 106.0), lat=(29.0, 32.0))


def describe(name, lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    print(f"\n=== {name} (n = {len(lon):,}) ===")
    for label, v in (("lon", lon), ("lat", lat)):
        q = np.percentile(v, [0, 1, 5, 50, 95, 99, 100])
        print(f"  {label}: min {q[0]:.3f}  p1 {q[1]:.3f}  p5 {q[2]:.3f}  "
              f"median {q[3]:.3f}  p95 {q[4]:.3f}  p99 {q[5]:.3f}  max {q[6]:.3f}")
    inside = (
        (lon >= CTU["lon"][0]) & (lon <= CTU["lon"][1])
        & (lat >= CTU["lat"][0]) & (lat <= CTU["lat"][1])
    )
    print(f"  inside Chengdu box    : {inside.sum():,} / {len(lon):,} "
          f"({100 * inside.mean():.2f}%)")
    # coarse global spread: fraction in each 30-degree longitude band
    bands = np.floor((lon + 180) / 30).astype(int)
    occupied = np.unique(bands)
    print(f"  30-deg lon bands used : {len(occupied)} of 12")


def main():
    with open(DATA / "2024-12-16-CTU_processed.geojson") as fh:
        gj = json.load(fh)
    lon = [f["geometry"]["coordinates"][0] for f in gj["features"]]
    lat = [f["geometry"]["coordinates"][1] for f in gj["features"]]
    describe("evaluation set", lon, lat)

    ref = pd.read_csv(DATA / "poi_without_ctu.csv", usecols=["lon", "lat"])
    describe("reference POI set", ref["lon"], ref["lat"])

    pts = pd.read_csv(DATA / "2024-12-16-CTU_processed_poi.csv",
                      usecols=["lon", "lat"])
    describe("kinematic baseline POIs", pts["lon"], pts["lat"])


if __name__ == "__main__":
    main()
