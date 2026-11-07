# Saved-output numerical-setting comparison

`beamfastsim.tssensitivity` compares two saved transport outputs against the same
reference. It never imports or runs transport, loads fields or geometry, or changes
transport defaults. Baseline and candidate are caller-assigned roles; the tool does
not infer or verify numerical settings or seeds.

```sh
python3 -m beamfastsim.tssensitivity \
  --reference reference.npz --baseline baseline.npz \
  --candidate candidate.npz --out comparison
```

All four arguments are required. Geometry/field arguments are rejected. The output
directory must not exist. Only `summary.json` and `settings.png` are created; no
companion logs, run records, metadata paths or checksum receipts are needed.
The Python API is `summarize(reference, baseline, candidate)`, accepting NPZ paths
or array mappings, with `plot_settings(result, path)` for the three-arm plot.

## Inputs and statistics

Inputs use the saved `tscompare`/`tsdiagnose` array formats, including terminal
`path`, `material_path` and scalar `seconds` for each transport output. The reference
must contain `event_id`, `parent_shard_ids` and `parent_shard_event_counts`.
Event IDs and parent inventories must be unique, counts must agree with the event
census, and every source event must belong to that census. Baseline and candidate
must have exactly the reference's unique source IDs. Their exposure columns, if
present, must be complete and match the reference; metadata-free saved outputs use
the reference exposure. Row, event and parent-inventory order may differ. Missing
source rows, mismatched shards, invalid terminal states and unresolved stock
endpoints are rejected.

There are no fixed run IDs, event totals, charge counts or shard counts. Parent-only
shards, zero-event inventory entries and shards with no detected outcomes remain
in the paired independent-shard calculation. Fractions use all original reference
inputs of the corresponding source charge, including losses, never detected-only
or parent-event denominators.

The summary includes stock comparisons for both outputs, candidate-minus-baseline
fractions and candidate/baseline ratios with paired-shard errors, DS species and
fate counts, fate transitions, capped IDs, muon overlaps, and momentum/elapsed-time
histograms. Histogram underflow and overflow counts are retained; the final edge
is inclusive. Undefined estimates are JSON `null`. A nonzero difference with zero
estimated SE is flagged, not reported as finite significance. The plot shows stock,
baseline and candidate spectra with candidate-minus-baseline pulls.

## Interpretation limits

Independent shards, not particles, determine errors. With few shards or sparse
positive-muon counts, precision claims are unsupported. Spectral diagonal chi2
includes yield, ignores bin covariance, excludes flow bins and is descriptive:
no calibrated p-value, equivalence test or significance from scanning bins.

One saved realization per setting mixes outcome variability and numerical effects;
it cannot isolate numerical bias or establish convergence. Equal seed labels do
not imply common-random trajectories when settings change RNG consumption. Joint
control changes cannot identify the contribution of one control. The comparison
does not select settings by proximity to stock.

## Completed joint-control comparison

The Run1A development comparison used 25,078 inputs across 100 shards and
500,000 parent events, seed 20261105. Field/material/loss caps changed jointly
from 2.5 mm / 0.25 mm / 5% to 1.25 mm / 0.125 mm / 2.5%; the saved baseline was
reused. These settings describe that comparison, not defaults of this tool.

| Endpoint | Baseline | Candidate | Stock | Candidate / baseline, paired SE |
|---|---:|---:|---:|---:|
| mu-minus | 1,088 | 1,073 | 1,098 | 0.98621 +/- 0.01985 |
| mu-plus | 8 | 9 | 15 | 1.125 +/- 0.437 |

Mu-minus source-fraction change was -0.12249 +/- 0.17769 percentage points.
Paired-shard momentum/time diagonal chi2 were 4.643/11 and 22.352/21 for
mu-minus; 3.347/3 and 6.351/8 for mu-plus (chi2 / occupied bins). Maximum absolute
bin pull was 1.56. Both settings had zero muon histogram flows,
live/path-limit/missing-field outcomes and DS pi-plus; each had 3 DS pi-minus.
There were 267 baseline-only and 252 candidate-only mu-minus endpoints.

No clear population instability was resolved at that sample size; this is not a
claim of convergence or precision agreement for sparse positives. These aggregate
results do not constrain the inputs accepted by the saved-output tool.

Synthetic regressions in `test_tssensitivity.py` cover exact joins, exposure
validation, paired uncertainties, original charge denominators, empty shards,
species/fates, flow bins, undefined SE, saved-only execution and no overwrite.
