"""Regenerate Fig. 2 (fidelity vs distance threshold) from the verified metrics.

Source of truth is ``results/regenerated_metrics.csv``, produced by
``regenerate_tables.py`` under a single 3D ECEF metric. Nothing is recomputed
here; the figure is drawn from those numbers and the values are printed so the
manuscript caption can be checked against them.

Outputs
-------
results/fidelity.png and results/fidelity.pdf

Deliberately written to results/ rather than over the manuscript's
thesis-v5/fidelity.pdf, so replacing the published figure is an explicit,
reviewable decision.

Usage:
    python fidelity_plot.py [--variant adf_adaptive]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "regenerated_metrics.csv"
OUT = HERE / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="adf_adaptive")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"{SRC} not found -- run regenerate_tables.py first")

    df = pd.read_csv(SRC)
    df = df[df["variant"] == args.variant].sort_values("Threshold_m")
    if df.empty:
        raise SystemExit(f"variant {args.variant!r} not present in {SRC.name}")

    print(f"variant: {args.variant}")
    print(df[["Threshold_m", "TT", "FP", "FN", "Precision", "Recall", "F1"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    x = df["Threshold_m"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x, df["Precision"] * 100, "o-", linewidth=1.2, label="Precision")
    ax.plot(x, df["Recall"] * 100, "s-", linewidth=1.2, label="Recall")
    ax.plot(x, df["F1"] * 100, "^-", linewidth=1.2, label="F1")
    ax.set_xlabel("Distance threshold (m)")
    ax.set_ylabel("Metrics (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(list(x))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"fidelity.{ext}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
