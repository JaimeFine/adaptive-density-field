# Latency audit — is the timing comparison defensible?

**Question:** the manuscript reports `ms/query` for ADF, KNN, brute force, `nprobe`
and `k` variants. Are those numbers comparable to each other?

## What is in the manuscript

| Where | Value | Source |
|---|---|---|
| ADF adaptive (`nprobe`=16, `k`=100) | 0.1145 ms/query | `adf_processor.py` (original run) |
| Brute force (`IndexFlatL2`) | 10.8184 ms/query | `brutal_force.py` (original run) |
| `nprobe` = 4/8/16/64/256 | 0.1255 / 0.1484 / 0.1596 / 0.2920 / 0.9571 | `nprobe.py` + `adf_processor.py` |
| `k` = 50/100/150 | 0.1435 / 0.1596 / 0.1724 | `k_value.py` |
| KNN `k` = 25/50/75/500 | 0.3671 / 0.3361 / 0.3499 / 0.4753 | `knn_processor.py` (original run) |

## Verdict

**Defensible as a single-machine, single-session measurement. Not portable, and
not statistically characterised.**

Reasons it is defensible:

1. **One machine, one period.** Every published latency came from the original
   authoring machine in the original session, so the *ratios* (notably ADF
   ≈ 0.1145 ms vs brute force ≈ 10.8184 ms, i.e. ≈ 94×) are meaningful.
2. **Same measurement convention.** All scripts use
   `total_time / total_query_points` with `time.perf_counter()`, where
   `total_query` counts all trajectory points processed. KNN and ADF therefore
   use identical conventions.
3. **Same per-call path.** Both `adf_processor.py` and `brutal_force.py` query
   one point at a time in a Python loop, so the ≈ 94× ratio reflects the index,
   not a difference in batching.
4. **The KNN fix does not invalidate its timing.** Changing the selection
   percentile (75 → 25) alters only the final threshold comparison on values
   that have already been computed; it does not change the neighbour search or
   the loop. The original KNN latency remains valid for the current code.

Reasons to be cautious:

1. **Machine-dependent absolute values.** A re-run on a different machine
   (2026-10) gave KNN 0.6537 / 0.5659 / 0.4812 / 0.7169 ms — roughly 1.4–1.9×
   larger. The published table therefore must **not** mix timings from two
   machines, which is why the original KNN latencies were retained rather than
   replaced by the re-run values.
2. **Single run, no repetitions.** There is no warm-up control, no repeated
   trials and no dispersion estimate, so no confidence interval can be quoted.
3. **Environment not recorded.** CPU, RAM, Python and FAISS versions for the
   original run were not captured.

## Recommendation

- Keep the published latencies, and state in the manuscript that timings are
  from a single machine and are reported as indicative ratios rather than
  portable absolutes.
- If a venue requires defensible timing, re-measure **every** variant in one
  session on one machine, with ≥5 repetitions and reported dispersion — and
  replace the whole column at once. Do not patch individual rows.

---

## Addendum 2026-10-12 — re-measurement attempted and abandoned

The re-measurement is **not published**, and no new timing workstream should be
opened. The manuscript keeps the historical figures plus the Limitations
caveats. This addendum records what was established so the attempt is not
repeated.

### Boundary asymmetry — verified by reading the scripts, not inferred

`adf_processor.py`, `nprobe.py` and `k_value.py` all start their timer **before**
the per-flight loop and **after** `index.train/add`. Their figure therefore
excludes index construction but includes coordinate conversion, the percentile
threshold, Python row construction, and one `print()` per flight.
`brutal_force.py` also excludes its `add`.

**`knn_processor.py` calls `knn.fit(pos)` *after* `start_bench`.** The k-NN
figure therefore includes building its neighbour structure while the ADF
figures exclude theirs, so the ADF-versus-k-NN latency comparison is not
like-for-like — independently of any session difference.

### Threading was never recorded

`faiss.omp_get_max_threads()` returns **16** on the authoring machine, and no
script pinned or logged it. Every published `ms/query` value is therefore a
parallel latency of unrecorded degree, which is not reproducible even on the
same machine. sklearn's `kneighbors` on a single query is effectively
single-threaded, giving a second asymmetry in the same comparison.

### Provenance correction

`nprobe.py` sweeps `[4, 8, 64, 256]` — it contains **no `nprobe=16` row**, so
the table's `nprobe=16` value (0.1596) does not come from it as the table above
states. `k_value.py` reports 0.1596 for `k=100` with an identical boundary, so
that is the likely source.

### The attempt, and why it was abandoned

A harness was built with pinned threads (OpenMP and BLAS set before the
numerical imports), an asserted reference dataset, index construction timed
separately, a matched query subset for the brute-force ratio, and a drift check
that measures the reference configuration **first and last** in the run.

It was aborted: the drift check returned **71.52%** — 0.4338 versus 0.1236
ms/query for the same configuration. The cause was the **CPU clock, not the
code**: the two passes performed identical work but consumed **31.4 s versus
10.1 s of CPU time**, which is only possible under frequency scaling. Two
orderings in the same run were also physically impossible (`nprobe=4` slower
than `nprobe=8`; `k=50` slower than `k=150`). The machine was on the **Balanced**
power plan and in active use.

The harness has since been removed; it had no other consumer. The useful
numbers it did produce are the CPU-time column above and the observation that
the clean rows were mutually consistent (`nprobe=8` < `nprobe=16` < `nprobe=64`;
`k=150` > `k=100`), so a stable machine would yield a coherent column.

### Consequence for the manuscript

The inconsistency is **disclosed rather than re-measured**. The Limitations
section already states that the figures come from single-machine
single-session runs without repetitions, that not every figure originates in
the same session so cross-session ratios are indicative only, and that the
measured scope differs by method. A sentence was added at the point of use in
the FAISS comparison, so a reader comparing the 0.1145 and 0.1596 rows for the
same configuration is told in the manuscript not to compare across tables.
