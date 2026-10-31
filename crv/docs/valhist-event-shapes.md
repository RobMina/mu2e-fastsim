# Conditional event-paired cluster shape diagnostics

`valhist.event_shapes` is a standalone, read-only analysis of two **complete**
`CrvRecoTableDumper` ROOT arms, such as A1 and E2E. It compares cluster-fill
shapes on the same generated events, including events with no clusters. It
writes a new JSON report and does not modify inputs or existing comparators.
There is **no physics acceptance flag or p-value**.

## Invocation

Run from the `mu2e-fastsim` umbrella with the existing local environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core:crv \
  ~/mu2e/.venv/bin/python3 -B -m valhist.event_shapes \
  /data/a1/valhist.root /data/e2e/valhist.root \
  --edges /data/analysis/cluster-edges.json \
  --replicas 999 --seed 12345 --out NEW_event_shapes.json
```

For multiple shards, supply a JSON list, not globs or positional directory
matching. Example `/data/analysis/paired-shards.json`:

```json
[
  {"ref": "/data/a1/shard-01.root", "cand": "/data/e2e/shard-01.root"},
  {"ref": "/data/a1/shard-02.root", "cand": "/data/e2e/shard-02.root"}
]
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core:crv \
  ~/mu2e/.venv/bin/python3 -B -m valhist.event_shapes \
  --pairs /data/analysis/paired-shards.json \
  --edges /data/analysis/cluster-edges.json \
  --replicas 999 --seed 12345 --out NEW_event_shapes_shards.json
```

Pair-list paths are relative to the **working directory**, not the JSON file.
Two positional files and `--pairs` are mutually exclusive. `--out` must not
exist. Exit 0 means the report was written, not that the arms agree. Invalid
inputs/options give exit 2; no partially calculated report is written.

`--edges` is a nonempty JSON object mapping observable names to exact lists
of finite, strictly increasing bin edges. There are deliberately **no default
physics edges**. A small example of its syntax, not a physics recommendation:

```json
{"npulses": [0, 1, 2, 3, 4]}
```

Edges are converted to float64 and recorded in the report. ROOT flow semantics
apply: underflow is below the first edge, bins are left-closed/right-open,
and a value **equal to the last edge is overflow**. Nonfinite cluster values
fail rather than disappearing into flow or being discarded.

## Observable definitions and source review

The implementation's allowlist is mapped to
`art/CrvFastSim/src/CrvRecoTableDumper_module.cc`:

| Observable | Dump branch / source definition |
| --- | --- |
| `pe` | `pe`: `CrvCoincidenceCluster::GetPEs()`, stored as float |
| `npulses` | length of `pulses`: `GetCrvRecoPulses().size()`; collection size, not unique-channel count |
| `start` | `start`: `GetStartTime()` |
| `duration` | `end - start`: `GetEndTime() - GetStartTime()` |
| `x`, `y`, `z` | `GetAvgHitPos()` coordinates, each stored as float |
| `sectortype` | `GetCrvSectorType()` |

Every cluster row contributes one unit-weight fill. No position-validity,
sector, pulse-quality, or timing cut is implicitly applied. Thus position
shapes include whatever coordinate values the dumper persists even if
`poscalc` is false. Time and position units follow the source (ns and mm).
There are no weighted-event or negative-weight semantics in this module.

Local `Offline/CRVDQM/src/CRVRecoDQM.cc` copies were reviewed under
`~/mu2e/remote/analysis/fastreco_p109`, `~/mu2e/remote/candidate`, and
`~/mu2e/remote/off_dqm`. In those copies, `fillClusters` fills `NPc` from
`GetCrvRecoPulses().size()`, `PEc` from `GetPEs()`, start-time histograms from
`GetStartTime()`, and position histograms only when the DQM configuration is
valid. These copies differ; their identity with the actual frozen production
binary is **not established** by this module. The production `CRVVAL_AREA`
backing tree was not available locally during implementation.

In particular, **dump `pe` is not asserted to reproduce a booked DQM PE
histogram**. Shared getter names do not establish booked-definition closure:
versions, cuts, conversion precision, and production configuration matter.
This minimal implementation supports **explicit edges only**. It does not
read or borrow `crvRecoDQM` histogram geometry, claim ROOT count closure, or
reinterpret a ROOT histogram as event data. Any future booked-geometry mode
must first check the actual row-derived bin counts, including both flows,
against the highest-cycle ROOT histogram separately in every arm/shard.

## Event validation and scope of completeness

For each file, the highest numeric cycles of `crvdump/events` and
`crvdump/clusters` are explicitly selected before class filtering; the
highest cycle must be a TTree. Older matching cycles cannot substitute for a
newer incomplete or incompatible one. The report records the selected keys,
all available cycles, entry counts, resolved paths, and sorted full-key hashes.

Each full identity is `(run, subrun, event)`, not just event number. Validation:

* Requires nonnegative integer IDs and cluster census counts.
* Sorts actual event keys and refuses duplicate rows.
* Requires identical full event populations within **each explicit pair**,
  including zero-cluster events; order within a file is irrelevant.
* Requires every cluster key to occur in its file's census, with row counts
  matching `ncluster` for every event.
* Checks `idx` is unique and in `[0, ncluster)` for each cluster collection.
* Refuses any reused full event key across shard pairs, even in copied or
  renamed files. A key occurring once in each arm is of course expected.
* Requires at least two generated events in the combined population.

There is no union-and-zero-padding recovery of missing events. Both arms
could omit the *same* generated events: two dumps alone cannot detect that.
Use externally verified generator identities and successful job completion,
as in the pilot completion checks, to establish actual completeness. Input
files must be stable during the read. The report does not replace source,
binary, conditions, or library provenance manifests.

## Shape covariance

For one region and observable, let `A_i` and `B_i` be vectors of bin counts
from paired generated event `i`, and `a_i`, `b_i` their regional totals. Define

```text
T_A = sum_i a_i                 T_B = sum_i b_i
p = sum_i A_i / T_A             q = sum_i B_i / T_B
delta = q - p
U_i = (B_i - b_i*q)/T_B - (A_i - a_i*p)/T_A
Cov(delta) = N/(N-1) * sum_i (U_i - mean(U)) (U_i - mean(U))^T
TV = sum_bins |delta| / 2
```

`U_i` is the event influence scaled by `1/N`. The covariance is for the
**estimated normalized bin difference**, not individual cluster counts. It
retains both within-event multiple-fill correlations and cross-arm pairing.
It is a plug-in, first-order event-sampling estimate. The covariance is
singular along the normalization constraint; no inverse, chi-square, ndf,
or significance claim is produced. Floating-point cancellation can leave
small negative roundoff eigenvalues; diagonal standard errors clip only
negative diagonal roundoff to zero, without modifying the reported matrix.

`inclusive` uses underflow, visible bins, overflow in that order.
`visible` discards flow and **renormalizes each arm to its own visible total**.
Zero regional totals make shape and covariance undefined, with explicit
status and null values. No pseudocounts are inserted. The sample size remains
all generated events, even when an event contributes nothing to the region.

## Centered conditional multiplier calibration and quantiles

Each replica draws independent standard Gaussian weights for the `N`
generated events. The weights are shared between the arms and centered:

```text
w_i = g_i - mean(g)
Z = sqrt(N/(N-1)) * sum_i w_i U_i
null TV replica = sum_bins |Z| / 2
uncertainty TV replica = sum_bins |delta + Z| / 2
```

`conditional_null_tv_quantiles` gives a zero-bin-difference **conditional
multiplier null reference**, calibrated to the observed paired influence
covariance. It is not a permutation/exchangeability test, not a simulation
of a fully specified physics null, and not a p-value. Equal aggregate shapes
can have nonzero conditional uncertainty if the paired event contributions
differ; exactly matching event contributions have zero influence.

`linearized_tv_uncertainty_quantiles` describes the corresponding shifted
linearized sampling distribution. `linearized_bin_difference_quantiles`
has shape `(quantile, bin)`. Quantile levels are recorded explicitly. These
are not certified confidence intervals, simultaneous bands, or percentile
resampling of complete events. Linearized draws need not stay inside the
probability simplex; their TV can exceed one and is deliberately not clipped.
The nonsmooth TV functional near zero, sparse data, finite replicas, and
first-order approximation all limit interpretation. The seed and replica
count are recorded; a fixed seed and input ordering reproduce the draws.
Changing shard order can change Monte Carlo draws, not the covariance or
point estimate. The same seed is used across observables/regions; no
cross-observable covariance or multiple-testing statement is supplied.

All uncertainty assumes independent generated events and conditions on the
**fixed secondary library, conditions, transport, and model**. Finite-library
uncertainty, systematic variations, and shared correlations between distinct
generated events are not included.

## Rate is separate from normalized shape

`total_cluster_rate` records total fills, fills per generated event, paired
absolute differences and their standard errors, and the paired relative
rate difference and its delta-method standard error. For the totals,

```text
SE(total_B - total_A) = sqrt(N * sample_variance(b_i - a_i))
R = total_B / total_A
SE(R) = sqrt(N * sample_variance(b_i - R*a_i)) / total_A
```

A zero reference total gives null relative quantities. These are cluster
fills per event, **not physical exposure-corrected rates**.
`visible_rate` and `flow_rate` expose regional fill changes separately for
each observable. The latter combines underflow and overflow; the inclusive
count vectors retain the two flow bins separately. Neither rate equality
nor normalized-shape equality implies the other.

## Memory and local regression

There are no Python per-event/cluster loops. Small loops over files,
observables, and multiplier replicas use vectorized numpy/awkward and SciPy
sparse operations. Per-event bin contributions are CSR matrices; influence
vectors are represented as a sparse difference plus a rank-two correction.
No dense `N x bins` influence/count matrix and no `replicas x N` multiplier
matrix is allocated. Only one observable is loaded across shards at a time;
only one replica's event weights is resident. Raw trees are read a file at a
time. Stored census keys/counts and CSR data scale with the combined sample.

Default hard limits are `--max-events 1000000` and
`--max-cluster-rows 5000000` **per arm across all shards**. Limits are checked
before reading table arrays. Raising them explicitly permits larger memory
use; this is not a constant-memory stream over arbitrary datasets. At most
256 visible bins per observable and 10000 replicas are permitted. Dense
workspace is `O(N)` plus `O(bins^2)` covariance and `O(replicas*bins)` draws;
sparse storage is `O(N + occupied event/bin cells)`. These limits bound major
numerical arrays, not every possible malformed ROOT jagged payload.

The focused synthetic suite covers dense-oracle covariance and centered
replica agreement, pairing versus independent-arm variance, multiplicity,
empty events/regions/arms, flows and exact last-edge handling, rate-only and
shape-only differences, full keys, census and collection-index failures,
actual highest-cycle selection, cross-shard duplicate events, strict edges,
nonfinite values, memory caps, CLI modes, input immutability, and exclusive
output creation. It also guards against densifying event/bin matrices.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core:crv \
  ~/mu2e/.venv/bin/python3 -B -m pytest -p no:cacheprovider \
  crv/tests/test_valhist_event_shapes.py -q
```

Measure local runtime on a representative shard before scaling a campaign;
no production runtime or physics numerical result is implied by synthetic
test performance.
