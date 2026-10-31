# Paired-shard histogram shape uncertainty

`valhist.paired_shapes` is a read-only, standalone complement to `valhist.coverage`
and `valhist.paired_rates`. It resamples **original shard pairs**, not bins or
individual fills. No independent-bin, Poisson or source-Sumw2 variance model is
used. This is an exploratory uncertainty analysis, not a physics acceptance test.

## Interface

From the `mu2e-fastsim` umbrella:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core:crv \
  python3 -B -m valhist.paired_shapes pairs.json \
  --out NEW_paired_shapes.json \
  --select '^crvRecoDQM/(PEc|NPc|X_extracted|Y_extracted|Z_extracted)$' \
  --count-key crvRecoDQM/PEc --count-key crvRecoDQM/NPc \
  --count-key crvRecoDQM/X_extracted --count-key crvRecoDQM/Y_extracted \
  --count-key crvRecoDQM/Z_extracted \
  --replicas 10000 --seed 12345 --batch-size 256 --confidence 0.95
```

Use a **JSON list**, not a glob, merged file or positional arm list:

```json
[
  {
    "shard_id": "001400_00000001",
    "ref": "/data/original/reference.001400_00000001.root",
    "cand": "/data/original/candidate.001400_00000001.root",
    "expected_events": 50000
  },
  {
    "shard_id": "001400_00000002",
    "ref": "/data/original/reference.001400_00000002.root",
    "cand": "/data/original/candidate.001400_00000002.root",
    "expected_events": 50000
  }
]
```

These are illustrative paths/counts, not a campaign result. Each row requires
`shard_id`, `ref`, and `cand`; `expected_events` is optional and must be a positive
integer. Unknown fields are errors. Paths are local, explicit and relative to the
JSON file if not absolute; URLs/globs are refused. IDs use `run_subrun` (or
`run:subrun`) and are canonicalized numerically. Rows are sorted by identity, not
paired by list position. At least two distinct shard pairs are required; two is a
technical minimum, **not** evidence of adequate bootstrap calibration. Reusing a
file/inode across shards is rejected. The same file in both arms of one shard is
allowed for exact-closure controls.

Options:

- `--select REGEX`: canonical histogram selection; default is the coverage
  module's CRV selection. An empty regex selects all objects, so event trees and
  unsupported objects then need explicit exclusions. Empty histogram selections
  are errors. A narrow selection defines a limited analysis scope, not full
  histogram coverage.
- `--alias OLD=NEW`: repeatable immediate `Validation/` child-prefix aliases,
  with the same longest-prefix, non-chaining rules as `coverage`.
- `--count-key EXACT_CANONICAL_PATH`: source-reviewed count-semantics override,
  including conservatively classified DQM histograms. No wildcards or automatic
  promotion. Unknown/unused overrides are errors; profiles remain unsupported.
- `--exclude-key EXACT_CANONICAL_PATH`: explicitly omit a selected unsupported,
  derived, or missing-in-an-arm object. Omissions are recorded; unused exclusions
  are errors. Exclusion never converts an object to counts, and cannot bypass
  alias collisions or TNamed metadata consistency. A key cannot be both excluded
  and overridden.
- `--replicas`, `--seed`, `--batch-size`, `--confidence`: defaults as in the command
  above. Replicas must be at least two, confidence strictly between zero and one.

`--out` must be a new file in an existing parent directory. Validation finishes
before writing; exclusive creation refuses overwrites. Exit 0 means a report was
written, **not agreement**; handled input/validation failures exit 2. ROOT files
are never written. Original/unmerged provenance remains the caller's
responsibility: lost flow content from historical merges cannot be recovered or
reliably detected.

## Identity and exposure evidence

Each actual ROOT input must independently identify its shard through either:

1. A basename containing an unambiguous `run_subrun` numeric token (run has at
   least four digits, such as `001400_00000001`); or
2. Its highest-cycle `crvdump/events` census containing integer `run`, `subrun`,
   `event` columns. All entries must match the declared run/subrun, with no
   duplicates. Empty censuses are rejected. Generic `valhist.root` filenames
   therefore work when this tree is present.

If both filename and tree identify a shard, both must agree. Filename evidence
is taken from the basename, not a convenient parent-directory label. If both
arms have censuses, sorted persisted event IDs must match exactly, including
zero-activity events; the report records highest cycles, counts and sorted-ID
SHA256 hashes. Expected counts, when supplied, must match every available
census. No per-event Python loop is used.

A histogram-only reference, including A0g, establishes **only run/subrun pairing
from its filename**, not event pairing or complete/equal exposure. Its supplied
expected count is recorded as `unverified_histogram_only`, never checked against
histogram fills or copied from the candidate's census. No event-paired error is
claimed in that case. Matching available event censuses still does not prove
completeness unless an external generated-event population has been established;
a matching count alone is weaker than matching an external population. The
bootstrap unit remains the shard even when event IDs match.

A separately copied dump/manifest is not automatically trusted as evidence for
a histogram ROOT file. There is no manifest-sidecar override in this version.
If the reference has neither a trustworthy run/subrun basename nor a census,
this mode refuses it rather than accepting the JSON row's declaration as proof.
Record external completion/provenance evidence separately. Filename checks do
not establish independence, authenticity, matching generator weights or exposure.

## Validation and count semantics

Highest numeric cycles are chosen **before class filtering** for every object,
including event trees and metadata. Every selected, non-excluded histogram must
exist in both arms and every shard. Exact ROOT classes (including storage type),
array shape, all finite strict-increasing axis edges, labels, histogram title and
axis titles must match globally. No approximate edge tolerance or rebinning is
used. All `TNamed` objects anywhere in the inputs are checked for equal inventory,
name and title across all arms/shards, independent of the histogram selection;
missing or changed binning-version records are errors. Their selected cycles are
recorded. No absent-in-both metadata/catalogue booking can be inferred: use the
separate coverage/catalogue audit for an independent booking inventory.

Profiles and unsupported selected objects are rejected, never normalized.
Coverage's conservative derived-name/title and DQM classification is reused;
ambiguous DQM histograms require exact reviewed count overrides. Other TH1/2/3
objects are merely `count_like`, **not certified event counts**. Arbitrarily named
derived maps cannot be identified universally; source review is still essential.
Negative/nonfinite bin contents, invalid edges and malformed/nonfinite/negative
source Sumw2 are fatal even for overridden count keys. Missing Sumw2 is recorded
and allowed because uncertainty comes from shard variation, not fill variance.
Zero-content shards are retained, not filtered out. Nonfinite arithmetic fails
rather than being reported as agreement.

## Estimands and resampling

For each histogram, each shard contributes its whole bin array. Let `a_s` and
`b_s` be the reference/candidate shard arrays, `A=sum_s a_s`, `B=sum_s b_s`.
Compute independently for:

- **inclusive**: all source flow-inclusive cells;
- **visible**: every axis in range.

The complement flow total is recorded; multidimensional corners count once.
Region arrays are flattened in NumPy C order, with explicit source flat indices
and flow-array shape for reconstruction. No merging, flow dropping or inferred
variance is performed.

The total ratio is `R=sum(B)/sum(A)` and relative difference is `R-1`.
It is a fill-weight-total ratio, not automatically an exposure-corrected physical
rate. Each arm's shape is normalized **separately**:

```text
p = A / sum(A)
q = B / sum(B)
D = q - p
TV = 0.5 * sum(abs(D))
```

A common multiplicative normalization change leaves this shape unchanged.
With `n` shards, each replica draws `n` pairs with replacement. Multinomial
multiplicities `w_s` are applied to **both arms**: `A*=sum_s w_s a_s` and
`B*=sum_s w_s b_s`. Totals, separately normalized shapes and TV are recomputed
in each replica. This preserves within-shard arm/bin correlations without
claiming within-shard fills are independent. Common draws are regenerated from
the same seed for every histogram/region, preserving their joint resampling
coupling as well. Seed, NumPy version, generator, batch size and replica count
are recorded. The implementation loops over files, histograms and matrix batches,
never events, bins or individual bootstrap draws. Temporary bootstrap matrices
scale as `batch_size * bins`; lower batch size for large multidimensional arrays.
Stored summaries scale as shards times bins plus replica count, not replicas
times bins.

Outputs include pointwise percentile intervals for the total ratio and TV, bin
fractions/differences, and per-bin centered-bootstrap RMS and bias:

```text
RMS_i = sqrt(mean((D*_i - D_i)^2))
bias_i = mean(D*_i - D_i)
```

RMS includes bootstrap bias; it is neither a source-Sumw2 error nor a
simultaneous confidence band. TV percentile intervals are descriptive bootstrap
intervals, not a test of the zero-TV boundary: use the separate centered diagnostic
with its limitations. No binwise independence or diagonal covariance is asserted.

### Centered TV null-calibration diagnostic

For every defined shape replica:

```text
T0* = 0.5 * sum(abs(D* - D_observed))
tail_plus_one = (1 + count(T0* >= TV_observed)) / (1 + valid_replicas)
```

The report records null quantiles (50/90/95/99%), mean/RMS, exceedances,
plus-one tail, its Monte Carlo resolution, and an approximate Monte Carlo
standard error. This evaluates a **centered bootstrap fluctuation norm**, not
an independent-Poisson statistic, binwise chi-square or exact randomization test.
The Monte Carlo error describes simulation noise only, not calibration error.
Centering does not produce a new physical nonnegative null histogram. Discrete,
sparse or low-shard samples can give degenerate/poorly calibrated distributions.

### Undefined replicas and multiplicity

If the reference total is zero, its ratio is undefined. A zero candidate with a
positive reference has ratio zero but undefined normalized shape. Shape requires
both totals positive. The report retains observed undefined statuses, `null`
quantities, separate zero-reference/candidate/both replica counts and valid
replica denominators. When some draws are undefined, intervals, bin RMS and null
tails explicitly describe **only the defined draws**. They have no claimed
unconditional coverage; do not silently reinterpret the surviving sample as the
requested replica population. All-zero regions are valid inputs with undefined
shape, not agreements.

Intervals/tails are unadjusted and pointwise. Histogram/region comparison counts
are recorded; bins and overlapping histograms are dependent. There is no
multiple-testing correction, family-wise acceptance threshold, pass flag or
blanket acceptance. Any confirmatory multiplicity policy must be chosen outside
this exploratory tool, not retrofitted by picking favorable histograms.

All uncertainty statements assume **exchangeable independent shard pairs,
conditional on the fixed secondary library and conditions**. Common library
reuse, calibration/model systematics, generator differences and finite-library
uncertainty are not estimated. Unequal shard exposure/composition can invalidate
exchangeability; review the declared/verified event counts and provenance first.

## Local tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core:crv \
  python3 -B -m pytest -p no:cacheprovider \
  crv/tests/test_valhist_paired_shapes.py -q
```

Synthetic tests cover normalization versus shape, exact preservation of correlated
pairs, direct multinomial-reference agreement, batch-size invariance, visible and
multidimensional flow behavior, zero/negative/nonfinite inputs, identity permutation,
event-census mismatch/duplicates/counts, highest histogram/tree cycles, metadata,
count overrides, explicit omissions, strict JSON, input immutability and CLI errors.
No campaign measurements or physics results are embedded in this document.
