# `results/` — what each file is and what it backs

Everything here is a generated artefact. Nothing in this directory is edited by
hand; each file is rewritten by the script named below.

## Metrics

| File | Produced by | Backs |
|---|---|---|
| `regenerated_metrics.csv` | `regenerate_tables.py` | Tables 1–5 and 7; Figure 2. One row per variant, 58 rows. The single source of truth for every number in the manuscript. |
| `regenerated_metrics_before.csv` | `regenerate_tables.py` | The 17-variant snapshot taken before the sweep was extended. Retained to show the extension reproduced those 17 rows exactly. |
| `poi_sets/` | `regenerate_tables.py` | One CSV per variant (58 files), holding the `lon`, `lat`, `alt` of the points that variant selected. The point sets behind each metrics row. |

## Figures

| File | Produced by | Manuscript |
|---|---|---|
| `fidelity.pdf`, `fidelity.png` | `fidelity_plot.py` | Figure 2 — fidelity versus tolerance |
| `poi_map.pdf`, `poi_map.png` | `poi_map_plot.py` | Figure 3 — spatial distribution of detections |
| `perflight_scatter.pdf`, `perflight_scatter.png` | `perflight_scatter_plot.py` | Figure 4 — paired per-flight F1 |

The plot scripts write here rather than over the manuscript's figure directory,
so that replacing a figure in the manuscript stays an explicit decision.

## Audit and diagnostic outputs

| File | Produced by | Contents |
|---|---|---|
| `baselines_audit_report.txt` | `diagnostics/baselines_audit.py` | Comparator implementations reviewed line by line against their sources |
| `sparse_stratified.csv` | `sparse_metric.py` | Metrics stratified by reference-set sparsity |

## Manuscript quality-control logs

| File | Produced by | Contents |
|---|---|---|
| `latex_artifact_scan.txt` | `tools/latex_artifact_scan.py` | Conversion-artefact scan of the manuscript source |
| `proofread_scan.txt` | `tools/proofread_scan.py` | Machine-detectable proofreading scan |

These two are build logs rather than experimental results, but they are kept
here because they are regenerated on every manuscript build and their presence
documents that the checks were run.

## Reproducing this directory

`regenerate_tables.py` requires `case-study/data/`, which is withheld (see
`../../DATA.md`). One figure — Figure 2 via `fidelity_plot.py` — can be
regenerated from the committed files alone, since it reads only
`regenerated_metrics.csv`.
