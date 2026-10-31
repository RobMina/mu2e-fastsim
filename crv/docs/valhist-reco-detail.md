# Retained A1 / hybrid reconstruction detail

`valhist.reco_detail` is a **read-only** follow-up to a completed
`valhist.reco_attribution` report. It does not launch reconstruction, response,
transport, builds, remote processes, or modify retained inputs. This diagnostic
locates differences in persisted pulse fits versus coincidence membership and
cluster calculations; it does not claim causal fractions or physics acceptance.

## Run / API

From the canonical CRV package, in the established analysis environment:

```bash
python3 -m valhist.reco_detail \
  --attribution /path/to/thread_b_reco_attribution_20261025 \
  --out /path/to/new_reco_detail.json
```

The attribution argument is a directory containing `attribution.json` and
`completion.json`. The output must be new, outside that directory, and not a
bound input. Existing files/symlinks are refused. No refresh or resume exists.

Python APIs:

* `create(attribution, out) -> dict`: accepted-report validation, per-shard
  comparison, before/after hash checks, and exclusive JSON creation.
* `compare_pair(ref, hybrid) -> dict`: low-level comparison of two local dump
  paths, with structural validation but **without accepted-provenance binding**.
  Use `create` for accepted production diagnostics.
* `numerical(a, b, pe=False) -> dict`: explicit mismatch statistics.

## Provenance gates

The report must be schema 1, completed, without changed inputs, and marked
hash-verified. Completion and report input maps must agree. **Every** completion
output and input hash is verified, including logs, other attribution products,
frozen sources, conditions and reference files—not just the two dumps read.
The report and original manifest must themselves be bound by completion.

Each shard's retained `valhist.fasttables reco` command identifies the A1 input;
that reference must have its original accepted hash in the report's
`input_sha256`. The hybrid path/hash must be bound by completion. There is no
current-hash fallback for an unbound reference. Hash bindings are checked again
before publication, including completion itself and this diagnostic's sources.
A failure publishes no success JSON. Hashes are integrity bindings, not signatures;
this does not protect against an adversary rewriting both report and completion.

The comparison reads **highest TTree cycles**, pairs the full `(run, subrun,
event)` tuples, retains zero-collection events, and checks finished event entries
against the accepted hybrid census. Event IDs and collection indices must be
nonnegative int64-compatible integers; duplicate keys, absent event identities,
inconsistent counts, noncontiguous/out-of-range indices, repeated membership
edges and invalid targets are rejected. No 20-bit or one-run label restriction
is used. Referenced waveform digis must have the pulse's channel.

## Matching and withholding

Pulse candidate keys are `(run, subrun, event, idx)`. They become usable for direct
numerical comparison only if **the whole event** has identical key sets,
`bar`, `sipm`, `seq`, and `wfidx` membership sets. Empty waveform membership is
rejected. Waveform order is not treated as identity; its membership is. Any key,
channel/sequence or waveform mismatch is counted, the affected event IDs are
reported, and its numeric pulse matching and cluster membership comparison are
withheld. There is no time/position/channel-nearest-neighbour rematching.

Clusters are paired by full event plus collection index **only in events with
equal cluster counts and identical sector order**. This is explicitly a
**collection-index diagnostic, not truth matching**, and not proof that the two
clusters correspond physically. Cluster scalar comparisons can remain eligible
when pulse matching is withheld; membership comparisons require both gates.

Membership is joined as structured rows `(run, subrun, event, cluster idx, pulse
idx)`, never packed integer labels. In JSON the final pulse index is called
`target`. `removed` and `added` contain the full edge sets plus sector; a removed
edge is not necessarily a lost pulse. Summaries describe edge-weighted PE/time
and SiPM counts. Pulses shared between clusters can contribute multiple edges.

## Output interpretation

Each shard reports:

* Full event and collection totals, per-event count differences and selected
  tree cycles.
* Pulse identity failure counts and withheld event IDs; numeric differences
  for `pe`, `time`, `seq`, `bar`, `sipm`, `flags` and common optional persisted
  fit fields (`peph`, `height`, `beta`, `chi2`, `ped`, `letime`, `pedfromdb`).
* Cluster alignment eligibility and membership removed/added/common edges.
* Every observed sector type, including sides: unpaired cluster totals and
  `poscalc` distributions; eligible `poscalc` transitions; cluster scalar shifts;
  unique eligible member-pulse shifts and removed/added membership counts.
* Four `cluster_strata`, crossing exactly identical persisted event pulses with
  changed membership. Their scalar deltas separate membership changes from
  cluster-value changes with fixed membership.
* `identical_pulses`: events with exact equality of all **common available**
  persisted scalar pulse fields and waveform membership, and the number of their
  eligible clusters/events with different membership. Optional fields present
  in only one arm are listed, not silently assumed equal.

A nonzero identical-pulse/different-membership count demonstrates a coincidence
membership difference with identical **persisted** pulse inputs. It does not
prove bitwise equality of pre-serialization doubles or internal algorithms.
The same limitation applies to fixed-membership cluster-value differences.
Zero such counts mean “not demonstrated on eligible rows,” not equivalence.
Withheld events are never interpreted as evidence of agreement.

Every numerical summary retains exact mismatch counts, finite and nonfinite pair
counts, signed hybrid-minus-A1 differences, explicit min/max, and quantiles
`0, .025, .5, .9, .95, .975, 1`. Mismatch-only and beyond-tolerance differences
are separate. NaN/Inf pairs are counted and excluded from numerical quantiles;
nonfinite values do not establish exact event pulse identity.

For serialized PE fields only, the documented envelope is
`2*eps(float32) + 2*eps(float32)*max(abs(A1), abs(hybrid))`. This is a deliberately
small serialization diagnostic, **not** an acceptance threshold or blind
`allclose`: even below-envelope differences remain in exact mismatch counts.
Other fields use zero tolerance; no implicit time or position tolerance exists.

Processing is vectorized NumPy/Awkward/Pandas within each shard. Python loops
cover files, fields, sector types and the four strata, not events/pulses.
Membership output can be large because removed/added sets are retained in full.
No pooling of shard quantiles is attempted; inspect the per-shard reports.

## Synthetic regression tests

```bash
python3 -m pytest -q -p no:cacheprovider \
  tests/test_valhist_reco_detail.py
```

Tests exercise multiple runs with repeated event numbers, large event IDs,
zero collections, highest cycles, row permutations, exact versus PE-envelope
mismatches, nonfinite values, channel/sequence/waveform withholding, cluster
sector/count gates, membership changes with identical pulses, invalid keys,
all bound file classes, mid-read drift, and unbound reference rejection.
