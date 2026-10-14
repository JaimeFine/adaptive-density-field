"""Explain selection-cardinality differences between ADF variants.

Each per-point output flags ``POI = (ADF >= P75(ADF within that flight))``.
If many ADF values tie exactly -- e.g. they underflow to 0 for a narrow
kernel -- the ``>=`` test selects far more than 25% of a flight, so that
variant's POI set is no longer cardinality-matched with the others, which
silently biases the comparison.

Usage:
    python check_selection.py [per_point_csv ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def describe(path: Path) -> None:
    df = pd.read_csv(path)
    if "POI" not in df.columns:
        print(f"{path.name}: no POI column, skipped")
        return

    value_col = next((c for c in ("ADF", "KNN") if c in df.columns), None)
    if value_col is None:
        print(f"{path.name}: no ADF/KNN column, skipped\n")
        return

    adf = df[value_col]
    total = len(df)
    selected = int((df["POI"] == 1).sum())
    n_zero = int((adf == 0).sum())

    per_flight = df.groupby("flight_id")
    fracs = []
    for _, g in per_flight:
        if len(g):
            fracs.append((g["POI"] == 1).mean())
    fracs = np.array(fracs)

    print(f"{path.name}")
    print(f"  points={total:,}  selected={selected:,} ({selected / total:.2%})")
    print(f"  {value_col} == 0: {n_zero:,} ({n_zero / total:.2%})")
    print(f"  per-flight share: min {fracs.min():.3f}  "
          f"median {np.median(fracs):.3f}  max {fracs.max():.3f}")
    print(f"  flights selecting >50%  : {int((fracs > 0.5).sum()):,} / {len(fracs):,}")
    print()


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        args = sorted(DATA.glob("ctu_*.csv"))

    for path in args:
        if not path.exists():
            path = DATA / path.name
        if path.exists():
            describe(path)
        else:
            print(f"{path}: not found\n")


if __name__ == "__main__":
    main()
