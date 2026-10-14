"""Threshold-based spatial matching between two POI point sets.

Python port of ``distance_sensitivity.R`` so the comparison harness can be
re-run without R installed. Both inputs are CSV tables containing
``lon``, ``lat`` (degrees) and ``alt`` (metres); points are converted to
WGS84 ECEF so the distances are true metric distances.

Definitions (identical to the R original):
    TT = points in A that have at least one point of B within <thr>
    FP = points in A that have no point of B within <thr>
    FN = points in B that have no point of A within <thr>

    precision = TT / (TT + FP)
    recall    = TT / (TT + FN)
    F1        = 2 * precision * recall / (precision + recall)

Usage:
    python metric_compare.py <a.csv> <b.csv> --thresholds 150 200 300 --out r.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

AXIS = 6378137.0
FLATTENING = 1 / 298.257223563
ECC2 = FLATTENING * (2 - FLATTENING)


def geodetic_to_ecef(lon, lat, alt):
    """WGS84 geodetic (degrees, metres) -> ECEF Cartesian (metres)."""
    lon = np.deg2rad(np.asarray(lon, dtype=float))
    lat = np.deg2rad(np.asarray(lat, dtype=float))
    alt = np.asarray(alt, dtype=float)
    n = AXIS / np.sqrt(1.0 - ECC2 * np.sin(lat) ** 2)
    return np.column_stack(
        (
            (n + alt) * np.cos(lat) * np.cos(lon),
            (n + alt) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - ECC2) + alt) * np.sin(lat),
        )
    )


def load_points(path, planar=False):
    df = pd.read_csv(path)
    missing = {"lon", "lat", "alt"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing required column(s) {sorted(missing)}")
    alt = np.zeros(len(df)) if planar else df["alt"]
    return df, geodetic_to_ecef(df["lon"], df["lat"], alt)


def compare(a_csv, b_csv, thresholds, planar=False):
    _, a_xyz = load_points(a_csv, planar=planar)
    _, b_xyz = load_points(b_csv, planar=planar)
    a_tree = cKDTree(a_xyz)
    b_tree = cKDTree(b_xyz)

    rows = []
    for thr in thresholds:
        a_hit = b_tree.query_ball_point(a_xyz, r=thr, return_length=True) > 0
        b_hit = a_tree.query_ball_point(b_xyz, r=thr, return_length=True) > 0

        tt = int(np.count_nonzero(a_hit))
        fp = int(a_xyz.shape[0] - tt)
        fn = int(np.count_nonzero(~b_hit))

        precision = tt / (tt + fp) if (tt + fp) else 0.0
        recall = tt / (tt + fn) if (tt + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        rows.append(
            dict(
                Threshold_m=thr,
                TT=tt,
                FP=fp,
                FN=fn,
                Precision=precision,
                Recall=recall,
                F1=f1,
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a_csv", help="primary point set (e.g. ADF output)")
    parser.add_argument("b_csv", help="reference point set (e.g. baseline)")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[100, 150, 200, 300, 400, 500],
    )
    parser.add_argument("--out", default=None, help="optional CSV output path")
    parser.add_argument(
        "--planar",
        action="store_true",
        help="ignore altitude and compare planimetric distances only",
    )
    args = parser.parse_args()

    result = compare(args.a_csv, args.b_csv, args.thresholds, planar=args.planar)
    print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
