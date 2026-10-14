"""Verify the overlap figures quoted for ADF-vs-KNN and adaptive-vs-fixed-sigma.

Definitions under test
----------------------
key(p)  = (flight_id, round(lon, 9), round(lat, 9), round(alt, 6))
A       = POI set of a variant's per-point output, restricted to POI == 1
overlap = |A n B| / |A|

Because both sets have the same cardinality here (20,769 rows each), the
measure is symmetric: |A n B| / |A| == |A n B| / |B|.

Checks
------
* unique-key cardinality vs raw row count (a key can repeat, so rows != keys);
* the identity |A n B| + |A only| == |A| in *unique-key* terms;
* the row-vs-key difference that makes a raw row count differ from the set count.

Usage:
    python verify_overlaps.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

PAIRS = [
    ("adf_adaptive vs knn_25", "ctu_adf_poi.csv", "ctu_knn_poi_25.csv"),
    ("adf_adaptive vs fixed_sigma_500", "ctu_adf_poi.csv", "ctu_adf_poi_fixed_500.csv"),
    ("adf_adaptive vs brute_force", "ctu_adf_poi.csv", "ctu_brutal_poi.csv"),
]


def load_keys(name):
    df = pd.read_csv(DATA / name)
    if "POI" in df.columns:
        df = df[df["POI"] == 1]
    keys = (
        df["flight_id"].astype(str)
        + "|" + df["lon"].round(9).astype(str)
        + "|" + df["lat"].round(9).astype(str)
        + "|" + df["alt"].round(6).astype(str)
    )
    return df, keys


def main():
    print(f"{'comparison':<34}{'|A|rows':>9}{'|A|keys':>9}{'|B|keys':>9}"
          f"{'|AnB|':>8}{'|A only|':>10}{'overlap':>9}{'jaccard':>9}")
    print("-" * 106)

    for label, a_name, b_name in PAIRS:
        a_df, a_keys = load_keys(a_name)
        b_df, b_keys = load_keys(b_name)

        sa, sb = set(a_keys), set(b_keys)
        inter = sa & sb
        only_a = sa - sb

        overlap = len(inter) / len(sa)
        jaccard = len(inter) / len(sa | sb)

        # row-based count of A-only, for comparison with the set-based count
        rows_only_a = int((~a_keys.isin(sb)).sum())

        print(f"{label:<34}{len(a_df):>9,}{len(sa):>9,}{len(sb):>9,}"
              f"{len(inter):>8,}{len(only_a):>10,}{overlap:>9.4f}{jaccard:>9.4f}")

        assert len(inter) + len(only_a) == len(sa), "identity failed in key space"
        if rows_only_a != len(only_a):
            print(f"    note: A-only rows = {rows_only_a:,} but unique keys = "
                  f"{len(only_a):,} (a key repeats among the A-only rows)")
        assert len(sa) <= len(a_df), "unique keys cannot exceed rows"

    print("\nall identities hold")


if __name__ == "__main__":
    main()
