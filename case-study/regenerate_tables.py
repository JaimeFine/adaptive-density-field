"""Regenerate every ADF comparison table under one consistent metric.

Background
----------
The published comparison table (``tab:validation-sensitivity``) and the
per-variant ablation tables were produced by ``distance_sensitivity.R``.
That script built sf objects with ``crs = 4326`` from a three-column
coordinate (lon, lat, alt), so altitude was not handled the way the paper's
methodology states ("all experiments were executed in the full 3D ECEF
coordinates system"). Its numbers cannot be reproduced from the repository
data with either a correct 3D ECEF metric or a purely planimetric one.

This script recomputes every comparison from the per-point ADF outputs using
a single, explicit metric: WGS84 ECEF Cartesian distance (metres).

Inputs are the per-point ADF outputs (``flight_id, lon, lat, alt, ADF, POI``);
rows with ``POI == 1`` form the extracted POI set.

Usage:
    python regenerate_tables.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from metric_compare import compare  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "results"
POI_SETS = OUT / "poi_sets"

REFERENCE = DATA / "2024-12-16-CTU_processed_poi.csv"

THRESHOLDS = [100, 150, 200, 250, 300, 350, 400, 450, 500]

# label -> per-point output file (rows with POI == 1 are the extracted POIs)
VARIANTS = {
    "adf_adaptive": "ctu_adf_poi.csv",
    "fixed_sigma_250": "ctu_adf_poi_fixed_250.csv",
    "fixed_sigma_250_tiesafe": "ctu_adf_poi_fixed_250_tiesafe.csv",
    "fixed_sigma_500": "ctu_adf_poi_fixed_500.csv",
    "fixed_sigma_750": "ctu_adf_poi_fixed_750.csv",
    "brute_force": "ctu_brutal_poi.csv",
    "nprobe_4": "ctu_adf_poi_4.csv",
    "nprobe_8": "ctu_adf_poi_8.csv",
    "nprobe_64": "ctu_adf_poi_64.csv",
    "nprobe_256": "ctu_adf_poi_256.csv",
    "k_50": "ctu_adf_poi_k=50.csv",
    "k_100": "ctu_adf_poi_k=100.csv",
    "k_150": "ctu_adf_poi_k=150.csv",
    "knn_25": "ctu_knn_poi_25.csv",
    "knn_50": "ctu_knn_poi_50.csv",
    "knn_75": "ctu_knn_poi_75.csv",
    "knn_500": "ctu_knn_poi_500.csv",
    # 2x2 factorial + mapping sweep (factorial_ablation.py)
    "fact_neither": "ctu_adf_poi_fact_neither.csv",
    "fact_score_only": "ctu_adf_poi_fact_score_only.csv",
    "fact_adaptive_only": "ctu_adf_poi_fact_adaptive_only.csv",
    "fact_both": "ctu_adf_poi_fact_both.csv",
    "fact_map_invshift": "ctu_adf_poi_fact_map_invshift.csv",
    "fact_map_exp": "ctu_adf_poi_fact_map_exp.csv",
    "fact_map_exp_wide": "ctu_adf_poi_fact_map_exp_wide.csv",
    # sigma0 sweep (factorial_ablation.py).  "sigma0_*" is the coupled ADF
    # operator (sigma_j = sigma0 / (s_j + eps)); "fixedbw_*" is the decoupled
    # form that bandwidth.py actually ran (sigma_j = sigma0).  sigma0_500 is
    # fact_both and fixedbw_500 is fact_score_only, so neither is repeated.
    "sigma0_100": "ctu_adf_poi_sigma0_100.csv",
    "sigma0_250": "ctu_adf_poi_sigma0_250.csv",
    "sigma0_1000": "ctu_adf_poi_sigma0_1000.csv",
    "sigma0_2000": "ctu_adf_poi_sigma0_2000.csv",
    "fixedbw_100": "ctu_adf_poi_fixedbw_100.csv",
    "fixedbw_250": "ctu_adf_poi_fixedbw_250.csv",
    "fixedbw_500": "ctu_adf_poi_fixedbw_500.csv",
    "fixedbw_1000": "ctu_adf_poi_fixedbw_1000.csv",
    "fixedbw_2000": "ctu_adf_poi_fixedbw_2000.csv",
    # nlist sweep (nlist_sweep.py), k=100 / nprobe=16 / sigma0=500 fixed.
    # Parity gate passed: nlist_4096 reproduces ctu_adf_poi.csv exactly, so
    # nlist_4096 and adf_adaptive must report identical metrics here.
    # Selection is bit-identical for nlist <= 4096 (Jaccard 1.0000 over a 64x
    # range); nlist=16384 diverges (n_poi 20,769 -> 32,105) because nprobe=16
    # then covers a far smaller share of the index.
    "nlist_64": "ctu_adf_poi_nlist_64.csv",
    "nlist_256": "ctu_adf_poi_nlist_256.csv",
    "nlist_1024": "ctu_adf_poi_nlist_1024.csv",
    "nlist_4096": "ctu_adf_poi_nlist_4096.csv",
    "nlist_16384": "ctu_adf_poi_nlist_16384.csv",
    # score-normalization + epsilon-floor sensitivity (normalization_sensitivity.py)
    "norm_raw": "ctu_adf_poi_norm_raw.csv",
    "norm_softplus": "ctu_adf_poi_norm_softplus.csv",
    "norm_minmax_1_2": "ctu_adf_poi_norm_minmax_1_2.csv",
    "norm_rank_1_2": "ctu_adf_poi_norm_rank_1_2.csv",
    "norm_softplus_bw": "ctu_adf_poi_norm_softplus_bw.csv",
    "norm_minmax_1_2_bw": "ctu_adf_poi_norm_minmax_1_2_bw.csv",
    "norm_rank_1_2_bw": "ctu_adf_poi_norm_rank_1_2_bw.csv",
    "norm_softplus_cal": "ctu_adf_poi_norm_softplus_cal.csv",
    "norm_minmax_1_2_cal": "ctu_adf_poi_norm_minmax_1_2_cal.csv",
    "norm_rank_1_2_cal": "ctu_adf_poi_norm_rank_1_2_cal.csv",
    "eps_0.0001": "ctu_adf_poi_eps_0.0001.csv",
    "eps_0.01": "ctu_adf_poi_eps_0.01.csv",
    "eps_0.1": "ctu_adf_poi_eps_0.1.csv",
    "eps_0.25": "ctu_adf_poi_eps_0.25.csv",
    # baseline suite (baselines.py)
    "base_kde_tree": "ctu_adf_poi_base_kde_tree.csv",
    "base_vbkde": "ctu_adf_poi_base_vbkde.csv",
    "base_vbkde_amp1": "ctu_adf_poi_base_vbkde_amp1.csv",
    "base_knn_kernel": "ctu_adf_poi_base_knn_kernel.csv",
    "base_nw": "ctu_adf_poi_base_nw.csv",
    "base_nw_tiesafe": "ctu_adf_poi_base_nw_tiesafe.csv",
}


def export_poi_set(label, filename):
    """Filter POI == 1 from a per-point output, write a clean POI table."""
    src = DATA / filename
    if not src.exists():
        print(f"  ! missing {src.name} -> skipped")
        return None

    df = pd.read_csv(src)
    if "POI" not in df.columns:
        print(f"  ! {src.name} has no POI column -> skipped")
        return None

    poi = df.loc[df["POI"] == 1, ["lon", "lat", "alt"]]
    if poi.empty:
        print(f"  ! {src.name} produced no POIs -> skipped")
        return None

    dst = POI_SETS / f"{label}.csv"
    poi.to_csv(dst, index=False)
    return dst, len(poi)


def main():
    POI_SETS.mkdir(parents=True, exist_ok=True)

    if not REFERENCE.exists():
        raise SystemExit(f"reference set not found: {REFERENCE}")

    frames = []
    for label, filename in VARIANTS.items():
        exported = export_poi_set(label, filename)
        if exported is None:
            continue
        path, n_poi = exported

        table = compare(str(path), str(REFERENCE), THRESHOLDS)
        table.insert(0, "variant", label)
        table.insert(1, "n_poi", n_poi)
        frames.append(table)
        print(f"  {label:<18} n_poi={n_poi}")

    result = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "regenerated_metrics.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    pivot = (
        result[result["Threshold_m"] == 200]
        .set_index("variant")[["n_poi", "TT", "FP", "FN", "Precision", "Recall", "F1"]]
    )
    print("\n=== 200 m threshold (3D ECEF) ===")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
