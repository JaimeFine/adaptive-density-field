# Attention in Geometry: An Adaptive Density Field (ADF) with FAISS-Accelerated Spatial Attention

Companion code for the manuscript of the same name.

> Preprint: https://arxiv.org/pdf/2601.06135

The manuscript was reviewed and rejected by IEEE TETCI (TETCI-2026-0303, 25 Sep
2026). This repository holds the corrected code, and every number in the revised
manuscript is generated from it rather than transcribed by hand.

This repository contains **code and results only**. The manuscript source
(`paper.tex`, the class files and the `.bbl`) lives outside it, in a sibling
directory of the same workspace. Scripts here write new figures into
`case-study/results/`, and a figure is copied into the manuscript only as a
deliberate step, never written over in place.

---

## What the code produces

| Manuscript artefact | Produced by |
|---|---|
| Table 1 — tolerance grid (150/200/300 m) | `case-study/regenerate_tables.py` via `metric_compare.compare` |
| Table 2 — reference scale σ₀ sensitivity | `case-study/factorial_ablation.py` (`sigma0_*`) |
| Table 3 — 2×2 component ablation | `case-study/factorial_ablation.py` (`fact_*`) |
| Table 4 — nprobe and k sweeps | `case-study/nprobe.py`, `case-study/k_value.py` |
| Table 5 — k-NN density-scan baseline | `case-study/knn_processor.py` |
| Table 6 — structural comparison | qualitative; no code |
| Table 7 — comparator family scores | `case-study/baselines.py` |
| Table 8 — scope of the claims | qualitative; no code |
| Fig. 2 — fidelity vs tolerance | `case-study/fidelity_plot.py` ← `results/regenerated_metrics.csv` |
| Fig. 3 — POI spatial distribution | `case-study/poi_map_plot.py` |
| Fig. 4 — paired per-flight F1 | `case-study/perflight_scatter_plot.py` |
| Score–density association (ρ ≈ +0.05) | `case-study/diagnostics/score_density_independence.py` |
| Normalization sensitivity (F1 0.6635–0.6655) | `case-study/normalization_sensitivity.py` |
| `nlist` failure mode (45 collapsed trajectories) | `case-study/nlist_sweep.py` |
| Paired per-flight intervals and tests | `case-study/diagnostics/perflight_stats.py` |
| Latency figures (0.1145, 10.8184, 0.3671 ms/query) | `adf_processor.py`, `brutal_force.py`, `knn_processor.py`, `baselines.py` |
| Appendix pipeline (α = ln 5 / k₉₅, λ = 1e-5, P75) | `appendix/processor.py` |

### Core modules

Two modules are imported by the rest and determine every downstream number:

* **`case-study/metric_compare.py`** — the canonical matching rule (WGS84 →
  ECEF, KD-tree radius query, precision/recall/F1). Eleven scripts import it,
  so a change here moves every table at once. It is a Python port of
  `distance_sensitivity.R`, so the harness runs without R installed.
* **`case-study/factorial_ablation.py`** — the 2×2 component ablation, the
  score→bandwidth mappings, and the shared constants
  (`SIGMA0 = 500`, `K = 100`, `EPS = 1e-6`, `PERCENT = 75`, `NPROBE = 16`,
  `NLIST = 4096`).

Known wrinkle: those constants are restated in `nlist_sweep.py` and
`normalization_sensitivity.py` imports them indirectly. They agree today, but
there is no single source of truth, so they can drift on the next edit.

### Verification scripts

Written during the October correction pass, which found and fixed an ENU
rotation-matrix error, a k-NN comparison inconsistency and a precision bug that
shifted several numbers:

`verify_metrics.py`, `verify_overlaps.py`, `characterize_diff.py`,
`check_selection.py`, `diagnose.py`, `sparse_metric.py`, `reselect_tiesafe.py`,
plus `diagnostics/`: `perflight_stats.py`, `baselines_audit.py`,
`component_pairs.py`, `coverage_check.py`, `dataset_stats.py`,
`score_suppression.py`.

---

## Reproduction

```bash
# 0. Environment (Python 3.11)
python -m venv .venv
.venv\Scripts\activate                       # Windows;  source .venv/bin/activate
pip install -r requirements.txt

# 1. Per-variant outputs
#    These read data/ relative to the current directory, so run them from
#    inside case-study/. They need case-study/data/, which is withheld (DATA.md).
cd case-study
python adf_processor.py            # ADF field, FAISS IVF retrieval
python bandwidth.py                # fixed-bandwidth / sigma_0 sweep
python nprobe.py                   # nprobe sweep
python k_value.py                  # k sweep
python nlist_sweep.py              # nlist sweep
python factorial_ablation.py       # 2x2 ablation + bandwidth mappings
python normalization_sensitivity.py
python baselines.py                # VBKDE, k-NN kernel, Nadaraya-Watson, tree KDE
python knn_processor.py            # plain k-NN density scan

# 2. Aggregate every variant into one metrics table
python regenerate_tables.py        # -> results/regenerated_metrics.csv

# 3. Figures (write to results/, copied into the manuscript deliberately)
python fidelity_plot.py            # Figure 2  -- runs without data/
python poi_map_plot.py             # Figure 3  -- needs data/
python perflight_scatter_plot.py   # Figure 4  -- needs data/

# 4. Manuscript checks (run after every edit to the manuscript source, which
#    lives outside this repository)
python ../tools/latex_artifact_scan.py <manuscript>/paper.tex
python ../tools/check_citations.py     <manuscript>/paper.tex <manuscript>/references.bib
python ../tools/proofread_scan.py      <manuscript>/paper.tex
```

`tools/run_in_dir.py` runs a script with a chosen working directory, because the
terminal used during development strips `cd`.

The plot scripts deliberately write to `results/` rather than over the
manuscript's figure directory, so replacing a figure stays an explicit decision.

---

## Repository layout

```
case-study/      the experiment harness
  diagnostics/   verification scripts (per-flight stats, baseline audit, ...)
  data/          third-party inputs and per-point outputs  (git-ignored; DATA.md)
  results/       aggregated metrics, figures, audit logs   (see results/README.md)
    poi_sets/    one CSV per variant: lon, lat, alt of the selected points
appendix/        appendix pipeline and technical report (see appendix/README.md)
tools/           manuscript scanners, used on every build
notebooks/       exploratory notebooks, pre-paper, kept for provenance
graphs/          figures as exported for the pre-revision submission (superseded)
drafts/          historical snapshots v0-v3            (git-ignored)
tempfolder/      local scratch                         (git-ignored)
```

At the repository root: `requirements.txt` (pinned environment), `DATA.md`
(what is withheld and what remains reproducible), `CITATION.cff` and
`LICENSE` (GPL-3.0).

### Kept deliberately rather than deleted

* `distance_sensitivity.R`, `distance_sensitivity_plot.R`, `poi_adf_mapper.R` —
  the original R implementation of the matching metric. Superseded by the Python
  port, but retained as the provenance of the metric.
* `nlist_failure_plot.py` — computes the `nlist` failure-mode statistics quoted
  in Section 5.5. Its figure was dropped from the manuscript because the prose
  states the result more precisely, but the script is the only reproducible
  record of those numbers, so it stays.
* `graphs/` — figure exports from the pre-revision submission. The live figures
  for the revised manuscript are produced into `case-study/results/` and copied
  into the manuscript from there.
* `LATENCY_NOTES.md` — provenance of every timing figure, including the
  withdrawn index-construction measurement and why it was withdrawn.

---

## Data

See **`DATA.md`** for the full statement, including which artefacts are
published in place of the data and which scripts remain re-runnable without it.

The trajectory data are supplied by a third party under terms that do not permit
redistribution, so `case-study/data/` is git-ignored and **not** included. The
manuscript reports this as a limitation. The pipeline expects three files there:

| File | Contents |
|---|---|
| `poi_without_ctu.csv` | the reference set — 2,243,080 scored points |
| `2024-12-16-CTU_processed_poi.csv` | kinematic reference labels — 20,501 |
| `2024-12-16-CTU_processed.geojson` | evaluation flights — 268 trajectories, 82,651 points |

Everything downstream of those three files is regenerable.

---

## Environment

Developed against Python 3.11 with `numpy`, `pandas`, `scipy`, `faiss`,
`scikit-learn` and `matplotlib`, in a virtual environment at `.venv/`
(git-ignored). Every version is pinned in **`requirements.txt`**, and the pins
match the environment in which every reported number was produced:

```bash
pip install -r requirements.txt
```

Retrieval was verified exact at the configuration used (`NLIST = 4096`,
`NPROBE = 16`): the accelerated and brute-force selections are identical. That
identity is what licenses treating the index as an accelerator rather than as
part of the operator, and it is a property of the configuration rather than of
IVF in general.

---

## Notes

* `progress.md`, `drafts/` and `tempfolder/` are development records, not part of
  the reproducibility package.
* Timing figures come from single-session measurements without repetitions. They
  are comparable within a run but not across runs — see `LATENCY_NOTES.md` and
  the Limitations section of the manuscript.
* Parameters quoted in the manuscript (σ₀ = 500 m, k = 100, ε = 1e-6, P75,
  nprobe = 16, nlist = 4096) match the code defaults in
  `factorial_ablation.py`.
