# All-charge numerical sensitivity (opt-in)

`python3 -m beamfastsim.tssensitivity` runs one prespecified full-development
candidate, retaining all 25,078 inputs in original order, 500,000 parent event IDs
and seed 20261105. Field/material/loss caps change jointly from
2.5 mm / 0.25 mm / 5% to 1.25 mm / 0.125 mm / 2.5%. The corrected baseline is
read from its saved NPZ and original transport log, never rerun. Default chains
and transport settings are not changed.

Pass `--reference`, `--baseline`, `--out` and all four geometry/field paths
(`--gdml`, `--field-config`, `--offline-root`, `--maps`) to run the one candidate.
With `--candidate <saved-npz>`, only saved output is summarized; geometry/field
arguments are rejected. The companion candidate `run.json` must identify the
prescribed inputs, seed and settings. Historical baseline material caps come from
the published command's defaults, not embedded NPZ metadata. Existing output
directories are refused. These commands describe the completed study, not a
request to repeat it.

Exact-ID joins retain all losses, parent-only shards, species and histogram flow
bins. Reference exposure must contain 500,000 parent events; saved fast exposure
arrays, when available, must match. Errors use independent paired shards and the
appropriate source-charge denominator, not POT. Output includes stock comparisons,
setting differences, fate transitions and three-arm spectrum/pull plots.

## Completed comparison (2026-11-06)

| Endpoint | Baseline | Joint tighter | Stock | Tighter / baseline, paired SE |
|---|---:|---:|---:|---:|
| mu-minus | 1,088 | 1,073 | 1,098 | 0.98621 +/- 0.01985 |
| mu-plus | 8 | 9 | 15 | 1.125 +/- 0.437 |

Mu-minus source-fraction change is -0.12249 +/- 0.17769 percentage points.
Paired-shard setting-difference momentum/time diagonal chi2 are 4.643/11 and
22.352/21 for mu-minus; 3.347/3 and 6.351/8 for mu-plus. Maximum absolute bin pull
is 1.56; no significance from scanning bins. Both settings have zero muon
histogram flows, live/path-limit/missing-field outcomes and DS pi-plus; each has
3 DS pi-minus.

No clear population instability is resolved at this sample size. This supports
keeping the baseline, not a claim of convergence or precision agreement of sparse
positives. Equal seeds are not common-random histories: 267 mu-minus baseline-only
and 252 tighter-only endpoints. Joint controls cannot isolate a numerical cause;
no setting is selected by closeness to stock. Diagonal chi2 omits bin covariance
and is descriptive, not a calibrated p-value or equivalence test.

Tighter transport cost is 398.133 s (62.989 entrances/s), one local worker,
excluding field load; tests ran during part of this job. The original baseline
benchmark and development headline remain authoritative. This module never reads
held-out data and is not a seed or settings scan.

Regression tests: `test_tssensitivity.py`, using synthetic references and mocked
transport to check one candidate call, no baseline rerun, exact exposure/identity,
paired uncertainties, no overwrite and invalid run-record rejection.
