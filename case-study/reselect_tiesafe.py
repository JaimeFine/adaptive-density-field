"""Audit per-flight selection cardinality and re-derive flags tie-safely.

Problem
-------
Every variant currently flags ``POI = (value >= P75(value) within flight)``.
When many values tie exactly -- which happens for a narrow kernel, where ADF
underflows to 0 -- that selects far more than 25% of a flight. The
``fixed_sigma_250`` variant shows the effect strongly (max per-flight share
1.000; 21,702 POIs instead of the reference 20,769), so its comparison is not
cardinality-matched with the others.

Rule applied
------------
For each flight take the reference variant's per-flight selected count ``K_f``
and keep exactly the ``K_f`` highest-value points, breaking ties by original
row order (stable ``mergesort`` -> deterministic). Cardinality parity with the
reference is then exact by construction.

What this does NOT change
-------------------------
Nothing about the ADF operator or the stored values: only the selection step
is re-derived, and only for variants whose per-flight counts differ from the
reference. The evaluation protocol (per-trajectory relative threshold at the
75th percentile, matched cardinality) is unchanged -- it is enforced exactly,
instead of approximately.

Usage:
    python reselect_tiesafe.py            # audit only
    python reselect_tiesafe.py --write    # also write *_tiesafe.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

REFERENCE_VARIANT = "ctu_adf_poi.csv"

VARIANTS = [
    "ctu_adf_poi.csv",
    "ctu_adf_poi_fixed_250.csv",
    "ctu_adf_poi_fixed_500.csv",
    "ctu_adf_poi_fixed_750.csv",
    "ctu_brutal_poi.csv",
    "ctu_adf_poi_4.csv",
    "ctu_adf_poi_8.csv",
    "ctu_adf_poi_64.csv",
    "ctu_adf_poi_256.csv",
    "ctu_adf_poi_k=50.csv",
    "ctu_adf_poi_k=100.csv",
    "ctu_adf_poi_k=150.csv",
    "ctu_adf_poi_base_kde_tree.csv",
    "ctu_adf_poi_base_vbkde.csv",
    "ctu_adf_poi_base_vbkde_amp1.csv",
    "ctu_adf_poi_base_knn_kernel.csv",
    "ctu_adf_poi_base_nw.csv",
]


def per_flight_counts(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=["flight_id", "POI"])
    return df.groupby("flight_id")["POI"].sum()


def reselect(path: Path, target_counts: pd.Series, value_col: str = "ADF") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_row"] = range(len(df))
    # stable sort: descending value, ties broken by original row order
    df = df.sort_values([value_col, "_row"], ascending=[False, True], kind="mergesort")
    df["POI"] = (
        df.groupby("flight_id").cumcount()
        < df["flight_id"].map(target_counts).fillna(0).astype(int)
    ).astype(int)
    # restore original ordering so the table stays comparable row-for-row
    df = df.sort_values("_row", kind="mergesort").drop(columns="_row")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write *_tiesafe.csv files")
    args = ap.parse_args()

    ref_counts = per_flight_counts(DATA / REFERENCE_VARIANT)
    ref_total = int(ref_counts.sum())
    print(f"reference: {REFERENCE_VARIANT}  flights={len(ref_counts):,}  "
          f"total={ref_total:,}\n")

    print(f"{'variant':<28}{'total':>8}{'Δ total':>9}{'flights≠K_f':>13}")
    print("-" * 60)

    mismatched = []
    for name in VARIANTS:
        path = DATA / name
        if not path.exists():
            print(f"{name:<28}{'missing':>8}")
            continue
        counts = per_flight_counts(path)
        total = int(counts.sum())
        aligned = counts.reindex(ref_counts.index).fillna(0).astype(int)
        n_diff = int((aligned != ref_counts).sum())
        print(f"{name:<28}{total:>8,}{total - ref_total:>+9,}{n_diff:>13,}")
        if n_diff:
            mismatched.append(name)

    print()
    if not mismatched:
        print("cardinality parity: all variants match the reference per flight.")
        return

    print(f"variants with per-flight mismatch: {', '.join(mismatched)}")
    for name in mismatched:
        if name == REFERENCE_VARIANT:
            continue
        fixed = reselect(DATA / name, ref_counts)
        total = int(fixed["POI"].sum())
        print(f"\n  {name}: reselected total = {total:,} "
              f"(reference {ref_total:,})  parity={'OK' if total == ref_total else 'FAIL'}")
        if args.write:
            out = DATA / (Path(name).stem + "_tiesafe.csv")
            fixed.to_csv(out, index=False)
            print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()
