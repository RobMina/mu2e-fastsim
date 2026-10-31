# Multi-shard finite-library ensemble diagnostic

`valhist.library_ensemble` is a read-only, manifest-driven analysis of a **balanced library × conditional draw × validation shard** table. A library realization is reused globally: it must be identical for every draw and every shard at its library index. Independent evaluation shards are never relabeled as independent libraries. This is the multi-shard extension of `library_uncertainty`, not a replacement for the existing one-shard diagnostic or a simulation driver.

No existing replay, library, plan, shell driver or report is modified. Old successful shard-11 bootstrap outputs can be listed unchanged alongside additional-shard outputs. Their directory names and historical seed formula are not parsed. A full report requires at least **two libraries, two draws per library, and two distinct shards**, with every Cartesian-product cell present. There is no partial-report mode or automatic dropping of failed cells.

## Exact CLI and manifest contract

From `crv/`, with the project environment and core available:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../core \
  python3 -m valhist.library_ensemble \
  --manifest /data/reports/library-cells.json \
  --out /data/reports/NEW-library-ensemble.json
```

Both arguments are required. The output's parent must already exist. Existing files, directories and dangling symlinks are refused. All certification and analysis finish before exclusive creation of the output file. The tool does not create report directories, submit jobs, build libraries, or discover inputs by glob. Files referenced by historical provenance must be locally accessible on the reporting host.

The input must be a JSON object with these fields; unknown fields are rejected:

| Field | Contract |
| --- | --- |
| `schema` | Required literal `"valhist.library_ensemble.v1"`. |
| `run` | Optional nonnegative integer, default `1400`; all persisted evaluation keys must have this run. |
| `cells` | Required nonempty list of the exact cell objects below. |
| `generator_denominator_files` | Optional **nonempty** list of existing ROOT paths; every supplied file's highest-cycle `dump/gen` is audited. |
| `generator_denominator_metadata` | Optional existing JSON path; it must contain a nonempty `gen` list of existing ROOT paths, all of which are audited. Other metadata fields are not interpreted. |

Every cell has **exactly** these fields:

| Cell field | Contract |
| --- | --- |
| `index` | Positive integer planned bootstrap index; maps to `bootstrap_` plus the index padded to at least four digits. No upper bound of eight. |
| `draw` | Positive integer draw label, with the same set of labels for every library and shard. Labels need not be consecutive. |
| `shard` | Positive integer matching persisted `subrun`; the same shard set occurs for every library/draw. |
| `directory` | Existing replay directory containing original `manifest.json`, `completion.json`, `a1/valhist.root`, and `e2e/valhist.root`. |
| `library_metadata` | Existing sealed `library.json` from `library_variants build`, shared by every cell at this index. The same-stem `.root`, metadata sidecar, and sibling sealed `plan.json` must exist. |
| `seed` | Explicit nonnegative integer matching the replay manifest. **Every cell's seed must be globally distinct**, including cells on different shards and at different library indices. |
| `expected_generated_events` | Explicit positive integer matching the transport census, completion and both complete persisted event tables. It must agree across cells of a given shard; different shards may have different exposures. |

JSON booleans and floating-point values are not integer identifiers or exposures. Every cell has a distinct replay directory. Top-level paths and cell paths resolve relative to the ensemble JSON directory; paths inside optional denominator metadata resolve relative to that metadata file. Paths inside generated build/plan/replay provenance remain their original absolute paths. No relocation or historical-provenance rewriting is performed.

Here is a complete minimal design (replace every placeholder directory with its actual existing path). These example seeds preserve the old index-1/index-2 shard-11 bootstrap seeds; the additional shard has independent seeds. The output paths need not follow these names.

```json
{
  "schema": "valhist.library_ensemble.v1",
  "run": 1400,
  "cells": [
    {"index": 1, "draw": 1, "shard": 11, "directory": "/data/replay/b1_d1_s11", "library_metadata": "/data/library/b1/library.json", "seed": 200101, "expected_generated_events": 50000},
    {"index": 1, "draw": 1, "shard": 12, "directory": "/data/replay/b1_d1_s12", "library_metadata": "/data/library/b1/library.json", "seed": 300101, "expected_generated_events": 50000},
    {"index": 1, "draw": 2, "shard": 11, "directory": "/data/replay/b1_d2_s11", "library_metadata": "/data/library/b1/library.json", "seed": 200102, "expected_generated_events": 50000},
    {"index": 1, "draw": 2, "shard": 12, "directory": "/data/replay/b1_d2_s12", "library_metadata": "/data/library/b1/library.json", "seed": 300102, "expected_generated_events": 50000},
    {"index": 2, "draw": 1, "shard": 11, "directory": "/data/replay/b2_d1_s11", "library_metadata": "/data/library/b2/library.json", "seed": 200201, "expected_generated_events": 50000},
    {"index": 2, "draw": 1, "shard": 12, "directory": "/data/replay/b2_d1_s12", "library_metadata": "/data/library/b2/library.json", "seed": 300201, "expected_generated_events": 50000},
    {"index": 2, "draw": 2, "shard": 11, "directory": "/data/replay/b2_d2_s11", "library_metadata": "/data/library/b2/library.json", "seed": 200202, "expected_generated_events": 50000},
    {"index": 2, "draw": 2, "shard": 12, "directory": "/data/replay/b2_d2_s12", "library_metadata": "/data/library/b2/library.json", "seed": 300202, "expected_generated_events": 50000}
  ]
}
```

Optional denominator declarations are **additional audit inputs**, not replacements for the actual denominator recorded in the sealed plan. Omitting them does not bypass the plan denominator audit. If both optional declarations are present, their union is audited; duplicate resolved paths are read once. An explicit unavailable, malformed, empty, or evaluation-overlapping denominator fails, rather than being ignored or downgraded to an unverified warning.

## Certification and held-out scope

1. Verify library metadata and plan SHA256 sidecars, the plan hash in build provenance, and the actual library ROOT hash. Check the bootstrap index, deterministic plan selections, selected ordered donor paths (including multiplicities), full-size generated exposure, source count, fixed denominator list and build-code identity. All libraries require the same plan, geometry, build code and denominator. Half libraries and arbitrary nominal metadata are unsupported. Duplicate library ROOT hashes at different indices are conservatively rejected; repeats are not credited as independent realizations.
2. Re-read all donor files in the common plan, not only those selected by one replica, using every highest-cycle `dump/particles` event key. Re-read every planned denominator's highest-cycle `dump/gen`. Check file hashes and exact inventories against the sealed plan. Full little-endian signed-64-bit `(run, subrun, event)` tuples are used; repeated particle rows within a file are deduplicated, but cross-file events and donor/denominator overlap fail. Source-code hashes recorded at build time are compared across builds, not against the current reporting checkout; this does not rebuild a library or retrospectively verify nominal code.
3. Require completed, undrifted replay records, completed arms and matching explicit seeds/exposures. Check actual output SHA256 and recorded output paths. Read each shard's retained transport, verify its hash and generated census, and require the same transport hash in every cell for that shard. Different shards may and normally do use different transport files.
4. Re-read full persisted event keys in both arms, require valid integer IDs, unique event rows, the exact expected population including zero-cluster events, agreement with transport and completion hashes, and exact run/subrun. `paired_rates.read` additionally checks cluster rows against the per-event cluster census. Different cells of a shard therefore use exactly the same events; different shards must have disjoint full event keys.
5. Match `campaign.logical_physics` identities across all cells, excluding the library, per-shard transport and analysis helpers. No shard-11 missing-hash exception is granted here. Matching inventories are historical evidence, not a rehash of inaccessible historical software/conditions or proof that unrecorded assets were captured. The existing strict completed-replay record is not regenerated.
6. Require one-dimensional count-histogram class and identical booking across all arms and cells for `PEc`, `NPc`, `X_extracted`, `Y_extracted`, and `Z_extracted`. Read original underflow, regular bins and overflow; require finite nonnegative integer counts and inclusive totals matching each arm's complete cluster census. No rebinning, histogram merging on disk, weighting or lost flow is accepted.
7. Check the union of evaluation events against **all persisted donor keys**, the plan denominator keys, and every explicitly supplied denominator file. Any overlap fails before publication. In particular, old denominator-sharing shards 1–10 cannot pass merely because their subrun labels differ from one another. Planned source audits alone store key hashes, not reusable full key arrays, so accessible source ROOT files must be re-read for exact evaluation disjointness.
8. Bind report inputs, sidecars, source files and ROOT outputs by SHA256, and rehash every bound input immediately before writing. The output contains the explicit cell evidence, common logical physics/build identities and source-audit inventories. Sidecars are accidental-change detection, not signatures; trusted production provenance is still required.

Reading uses NumPy vectorization for event keys and covariance; loops are over small cell/file/histogram inventories. Exact source key arrays are retained in memory, so memory scales with the persisted donor and denominator populations. Production audit runtime is not inferred from small synthetic tests.

## Pooling and nested method of moments

For library `l`, conditional draw `d`, shard `s`, let `C_lds` be cluster count, `N_s` generated exposure, and `H_ldsb` original inclusive histogram counts (including both flow bins).

- Per-shard rate: `C_lds / N_s`.
- Pooled rate for **one** library/draw realization: `sum_s C_lds / sum_s N_s`.
- Per-shard shape: `H_ldsb / sum_b H_ldsb`.
- Pooled shape: `sum_s H_ldsb / sum_sb H_ldsb`.

Thus normalized shard shapes are **not averaged**, even with exposure weights; counts are summed first. The number of independent library realizations stays `L`, not `L*S`. A draw label denotes the whole vector of shard-specific random runs at that draw. It does not require or imply the same seed on different shards.

For each scalar or bin observable `Y_ld`, the same nested method of moments is applied **after pooling**, and separately on each shard:

```text
library_mean_l = mean_d Y_ld
within = mean_l Var_d(Y_ld), with ddof = 1
between_means = Var_l(library_mean_l), with ddof = 1
library_variance_noise_subtracted = between_means - within / D
```

The report retains library means, the grand mean, conditional-draw variance, library-mean variance and the signed noise-subtracted component. It adds a negative-component flag and `sqrt(max(component, 0))` as a separately labeled nonnegative SD display. It does **not** clip the estimate, divide it by the number of libraries and call it the full-library standard error, or subtract validation-shard fluctuations from library spread.

Calculations cover each arm plus paired **`e2e_minus_a1`**, subtracting within a cell before estimating variance. This preserves within-cell arm correlations. It reports no ratio covariance or ratio hypothesis test.

### Cross-shard covariance

For every scalar observable, and for each shape bin separately, the shard vector is kept intact. Conditional-draw covariance is the library average of within-library sample covariance across draws. Library-mean covariance is the sample covariance of shard-vector library means. The signed library covariance is their difference with the former divided by `D`.

All three matrices are retained; the final two axes are `(shard, shard)` and a shape bin adds a preceding bin axis. This preserves common-library cross-shard dependence and can expose accidental treatment of shards as new libraries. For rates, exposure-weighted contraction of this matrix reproduces the directly pooled variance. **Do not** apply that linear contraction to normalized shapes with variable cluster totals: pooled shape components are computed directly from the pooled counts. No cross-bin or cross-observable covariance matrix is supplied. Small-ensemble covariance can be indefinite; no PSD projection conceals that.

### Empty histograms

A zero-total histogram has undefined shape, not a zero shape. Raw counts remain available and its normalized values are JSON `null`. If any cell of a particular per-shard or pooled shape is undefined, that scope has `status: "undefined_zero_histogram_total"` and `components: null`; there is no incomplete-case fitting. A shard may be empty while its pooled histogram is defined. Cross-shard shape covariance is undefined if any participating cell is undefined. Zero-cluster **rates** remain defined. Output serialization disallows NaN/Infinity.

## Output and limitations

- `axes` contains sorted library indices, draw labels and shard IDs. `array_order` labels raw array axes.
- `raw_counts[arm]` retains cluster counts and each inclusive histogram as `[library, draw, shard, optional_bin]`; `pooled_counts[arm]` sums the shard axis.
- `generated_events_per_shard` and `pooled_generated_events` are exposures for **one** library/draw realization, not multiplied by repeated evaluations.
- `metrics[arm_or_difference][observable]` retains `values_per_shard`, `values_pooled`, pooled/per-shard component reports and cross-shard covariance.
- `histogram_bookings`, `provenance`, `physics`, `build_identity`, `source_audit`, `input_sha256` and `limitations` retain the interpretation and audit trail.

This remains a small-ensemble **conditional diagnostic**, not confidence coverage, a full uncertainty budget, a calibrated DQM acceptance test, or an estimate of unconditional validation-population variance. Whole-file bootstrap assumes exchangeable donor blocks; overlapping bootstrap donor selections are intentional, not proof of independent underlying donor datasets. Distinct seeds are necessary declarations, not empirical proof of independent pseudorandom streams. Tier1b/fast-response randomness is varied; fixed stock A1 random streams and conditions are not. Denominator uncertainty is fixed, not propagated, even when its events are demonstrably held out. Quantile-edge refitting remains part of the library variation.

Persisted donor disjointness says nothing about unpersisted generated donors or common production conditions. There is no arbitrary-library, half-library, unbalanced-design, common-random-seed, weighted-histogram, or missing-source-file fallback. Small minima make the decomposition possible, not reliable. Negative variance components and non-PSD covariance matrices remain visible. No result is extrapolated to unlisted shards or the full campaign.

## Local validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../core \
  python3 -m pytest -p no:cacheprovider -q \
  tests/test_valhist_library_ensemble.py
```

Tests use small synthetic ROOT sources and complete replay fixtures, never builds, remote data or production jobs. They cover balanced-design gates, explicit shard-specific seeds, old-style replay reuse, exposure/count pooling rather than normalized-shape averaging, flow preservation, nested covariance and signed negative estimates, undefined shapes, exact high-width event IDs, persisted duplicates, fractional-ID rejection, source/output drift, metadata/plan seals, matched physics/bookings, and donor/denominator evaluation overlap including explicitly supplied denominator metadata.
