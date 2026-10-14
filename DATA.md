# Data availability

The trajectory data used in this study were supplied by a third party under
terms that do not permit redistribution. `case-study/data/` is therefore
git-ignored and is **not** part of this repository. The manuscript states this
as a limitation.

This file says exactly what is withheld, what is published in its place, and
which parts of the pipeline a reader can therefore re-run unaided.

## Withheld — `case-study/data/`

The pipeline expects three input files plus the per-variant selection outputs:

| File | Contents | Approx. size |
|---|---|---|
| `poi_without_ctu.csv` | reference set — 2,243,080 scored points | 124 MB |
| `2024-12-16-CTU_processed_poi.csv` | kinematic reference labels — 20,501 points | small |
| `2024-12-16-CTU_processed.geojson` | evaluation flights — 268 trajectories, 82,651 positions | 25 MB |
| `ctu_*.csv` | per-variant selection outputs, written by the processor scripts | ~250 MB total |

None of these are redistributable. This is the constraint recorded in the
manuscript's Data availability statement.

## Published in place of the data — `case-study/results/`

Two artefacts are committed and are sufficient to inspect and check every
number that appears in the manuscript:

* **`results/regenerated_metrics.csv`** — the aggregated metrics table, one row
  per experimental variant (58 rows). This is the single source of truth for
  Tables 1–5 and 7 and for Figure 2. Every value quoted in the manuscript can be
  read directly from this file.
* **`results/poi_sets/`** — 58 CSVs, one per variant, each holding the `lon`,
  `lat`, `alt` of the points that variant selected. These are the point sets
  behind each row of the metrics table.

`results/regenerated_metrics_before.csv` is retained as well: it is the
17-variant snapshot taken before the sweep was extended, and it documents that
the extension reproduced the original 17 rows exactly.

## What can be reproduced without the withheld data

| Script | Output | Needs `data/`? | Re-runnable here? |
|---|---|---|---|
| `fidelity_plot.py` | Figure 2 | no — reads `results/regenerated_metrics.csv` | **yes** |
| `poi_map_plot.py` | Figure 3 | yes — reference labels | no |
| `perflight_scatter_plot.py` | Figure 4 | yes — via `diagnostics/perflight_stats.py` | no |
| `adf_processor.py` and the other processors | `data/ctu_*.csv` | yes | no |
| `regenerate_tables.py` | `results/regenerated_metrics.csv` | yes | no |
| `verify_metrics.py` | consistency checks | yes | no |

In short: **the reported values are published and independently inspectable, but
the selection step cannot be re-derived from scratch here.** Anything requiring
the reference set or the reference labels stops at the data boundary. This
distinction is stated rather than blurred, because a reader who cannot obtain
the data should still know precisely how far reproduction is possible.

## Re-running the full pipeline

Given the withheld inputs in `case-study/data/`, run the processors first (each
writes its own `data/ctu_*.csv`), then aggregate:

```bash
cd case-study
python adf_processor.py
python bandwidth.py
python nprobe.py
python k_value.py
python nlist_sweep.py
python factorial_ablation.py
python normalization_sensitivity.py
python baselines.py
python knn_processor.py
python regenerate_tables.py        # -> results/regenerated_metrics.csv
```

Note that the processor scripts resolve `data/` relative to the **current
working directory**, so they must be run from inside `case-study/`. The
aggregation and figure scripts resolve paths relative to their own location and
run from anywhere. `../tools/run_in_dir.py` exists to make the working directory
explicit.

## Requesting access

Requests for the underlying trajectory data should be directed to the data
provider. The authors cannot redistribute it on the provider's behalf.

## A note on `results/poi_sets/`

These files contain coordinates of points selected by the pipeline from the
third-party dataset. They are derived outputs, not the raw dataset, and they are
published so that the selection results can be audited. If the provider's terms
are read as extending to derived point sets, these files should be withdrawn;
they are isolated in one directory so that this can be done cleanly.
