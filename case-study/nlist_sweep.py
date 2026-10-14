"""nlist sweep for R1.4: is the IVF partition size binding?

Referee R1.4 asked why 4,096 partitions and a 500 m scale.  The 500 m half is
answered by the coupled sigma0 sweep already in the manuscript.  This script
answers the nlist half, which had never been swept.

Why it needs its own run
------------------------
nprobe and k are search-time parameters, so they can be swept without touching
the index.  nlist is an INDEX-CONSTRUCTION parameter: every value requires a
fresh k-means train plus add over the 2,243,080-point reference set.  That cost
is why this sweep was never done.

Configuration held fixed
------------------------
k = 100, nprobe = 16, sigma0 = 500 m, per-trajectory P75 threshold, float32
ECEF positions -- identical to the published configuration in every respect
except nlist.

Parity check
------------
nlist = 4096 IS the published configuration, so its output must reproduce
`ctu_adf_poi.csv` exactly.  If it does not, this harness differs from the
original pipeline and no other row is comparable.  The script reports the
check explicitly rather than assuming it.

Output
------
data/ctu_adf_poi_nlist_{n}.csv, one per value, in the canonical per-point
format (flight_id, lon, lat, alt, ADF, POI), plus a summary of selection
overlap against the published nlist = 4096 selection.

The five values that answer R1.4 are 64, 256, 1024, 4096 and 16384, the range
the referee specified.

Timing
------
Wall-clock elapsed time is reported alongside process CPU time.  Wall time
includes any period for which the machine was suspended, so the two must agree
before a build time is used as a measurement: a run whose wall time far exceeds
its CPU time was interrupted, and its duration measures the interruption rather
than the work.  (`time.perf_counter()` is documented to include elapsed sleep,
so a closed laptop lid inflates it.)  A re-measurement of one value reuses the
saved nlist = 4096 output as the overlap reference.

Usage
-----
    python nlist_sweep.py              # all five values
    python nlist_sweep.py 16384        # re-measure one value only
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
NLIST_VALUES = (64, 256, 1024, 4096, 16384)
K = 100
NPROBE = 16
SIGMA0 = 500.0
EPS = 1e-6
PERCENT = 75

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


def overlap_keys(df):
    """Unique-key overlap, matching the convention used elsewhere in the study."""
    return set(zip(
        df["flight_id"],
        np.round(df["lon"].to_numpy(), 9),
        np.round(df["lat"].to_numpy(), 9),
        np.round(df["alt"].to_numpy(), 6),
    ))


def main() -> int:
    ref = pd.read_csv(DATA / "poi_without_ctu.csv")
    pos = np.vstack([
        geodetic2ecef(lon, lat, alt)
        for lon, lat, alt in ref[["lon", "lat", "alt"]].to_numpy()
    ]).astype("float32")
    scores_all = ref["poi_score"].to_numpy()

    with open(DATA / "2024-12-16-CTU_processed.geojson") as fh:
        track = json.load(fh)

    flights: dict = {}
    for feature in track["features"]:
        fid = feature["properties"]["flight_id"]
        flights.setdefault(fid, []).append(feature["geometry"]["coordinates"])

    tracks = {}
    for fid, coords in flights.items():
        coords = np.array(coords)
        tracks[fid] = (
            coords,
            np.vstack([
                geodetic2ecef(lon, lat, alt) for lon, lat, alt in coords
            ]).astype("float32"),
        )

    values = tuple(int(a) for a in sys.argv[1:]) or NLIST_VALUES

    print(f"reference POIs : {len(pos):,}")
    print(f"flights        : {len(tracks)}")
    print(f"fixed          : k={K}, nprobe={NPROBE}, sigma0={SIGMA0}, P{PERCENT} threshold")
    print(f"varying        : nlist in {list(values)}\n")

    selections = {}
    for nlist in values:
        c0, t0 = time.process_time(), time.perf_counter()
        quantizer = faiss.IndexFlatL2(3)
        index = faiss.IndexIVFFlat(quantizer, 3, nlist)
        index.train(pos)
        index.add(pos)
        index.nprobe = NPROBE
        build_wall = time.perf_counter() - t0
        build_cpu = time.process_time() - c0

        c1, t1 = time.process_time(), time.perf_counter()
        rows = []
        for fid, (coords, track_ecef) in tracks.items():
            _d, idx = index.search(track_ecef, K)
            values = np.empty(len(coords))
            for j, x in enumerate(track_ecef):
                nb = pos[idx[j]]
                sc = scores_all[idx[j]]
                sigma = SIGMA0 / (sc + EPS)
                quad = np.sum((nb - x) ** 2 * (1.0 / sigma ** 2)[:, None], axis=1)
                values[j] = float(np.sum(sc * np.exp(-0.5 * quad)))

            threshold = np.percentile(values, PERCENT)
            mask = values >= threshold
            for (lon, lat, alt), v, m in zip(coords, values, mask):
                rows.append([fid, lon, lat, alt, v, int(m)])

        out = pd.DataFrame(rows, columns=["flight_id", "lon", "lat", "alt", "ADF", "POI"])
        path = DATA / f"ctu_adf_poi_nlist_{nlist}.csv"
        out.to_csv(path, index=False)
        sel = out[out["POI"] == 1]
        selections[nlist] = sel
        n_poi = int(len(sel))
        eval_wall = time.perf_counter() - t1
        eval_cpu = time.process_time() - c1
        print(f"nlist={nlist:<6} build {build_wall:6.1f}s wall / {build_cpu:6.1f}s cpu  "
              f"eval {eval_wall:5.1f}s wall / {eval_cpu:5.1f}s cpu  "
              f"n_poi={n_poi:,}  -> {path.name}")
        if build_wall > 2 * build_cpu + 60:
            print("        ! wall time far exceeds CPU time: the machine was "
                  "suspended or the process was descheduled. Do NOT use this "
                  "build time as a measurement; re-run it.")

    # ------------------------------------------------------------- parity --
    base = DATA / "ctu_adf_poi.csv"
    if 4096 in selections and base.exists():
        pub = pd.read_csv(base)
        pub_sel = pub[pub["POI"] == 1]
        mine = selections[4096]
        keys_pub = overlap_keys(pub_sel)
        keys_mine = overlap_keys(mine)
        same = keys_pub == keys_mine
        print(f"\n=== parity vs the published nlist=4096 output ===")
        print(f"  published n_poi = {len(pub_sel):,}   swept n_poi = {len(mine):,}")
        print(f"  unique-key selections identical : {same}")
        if not same:
            print("  *** MISMATCH: this harness differs from the original pipeline. ***")
            print("  *** No other row of this sweep is comparable until resolved. ***")
    else:
        print("\n! nlist=4096 not run in this pass; parity check skipped")

    # ------------------------------------------------------- overlap table --
    # Reference selection comes from this run when available and from the saved
    # output otherwise, so a single-value re-measure still reports overlap.
    ref_sel = selections.get(4096)
    if ref_sel is None:
        ref_path = DATA / "ctu_adf_poi_nlist_4096.csv"
        if ref_path.exists():
            ref_df = pd.read_csv(ref_path)
            ref_sel = ref_df[ref_df["POI"] == 1]
    if ref_sel is None:
        print("\n! no nlist=4096 selection available; overlap table skipped")
        return 0

    print(f"\n=== selection overlap against nlist=4096 (unique keys) ===")
    base_keys = overlap_keys(ref_sel)
    for nlist in values:
        sel = selections.get(nlist)
        if sel is None:
            continue
        keys = overlap_keys(sel)
        union = base_keys | keys
        jac = len(base_keys & keys) / len(union) if union else float("nan")
        print(f"  nlist={nlist:<6} n_poi={len(sel):>6,}  "
              f"overlap={len(base_keys & keys):>6,}/{len(base_keys):,} "
              f"({len(base_keys & keys) / len(base_keys) * 100:5.1f}%)  Jaccard={jac:.4f}")

    print("\nNext: register these files in regenerate_tables.py VARIANTS, then run "
          "the pooled metrics and the paired per-flight comparison under BOTH "
          "aggregations, as the plan requires for any remaining sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
