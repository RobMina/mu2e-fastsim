# Retained replay step-timing census

`python3 -m crvfastsim.veto.step_timing_audit` reads existing raw replay-v1 files
without generating photons, rerunning response or changing the fixed-stock sample.
It is the step-level prerequisite to the full optical timing experiment in
`veto-continuation.md`, not a substitute for that experiment. No F value, finite-library
error or promotion certificate is produced. Replay-v2 remains rejected.

```bash
python3 -m crvfastsim.veto.step_timing_audit \
  /data/replay/block0/mixed.root /data/replay/block1/mixed.root \
  /data/replay/block2_fullinventory/mixed.root \
  --counters /release/Offline/Mu2eG4/geom/crv_counters_run1a_v01.txt \
  --stock-source /release/Offline/CRVResponse/src/CrvPhotonGenerator_module.cc \
  --out /data/new_step_timing_audit
```

The stock source must match the reviewed v13_39_00 SHA256
`1b5aef320f6f344d607fbf3c18e92b075b9d4daed84532d61ef78938dccd826e`.
The counters file must match SHA256
`e70a819bb0fe1a3a4c7253b9b32e87dace8f68de27aa82f054d9c3cdfc11ec57`;
sector names alone do not establish the bar-to-sector mapping. Every bar must
resolve, and all T1/T2 sectors in this Run1A configuration are included. No top-only or previously
successful-channel selection is applied. The audit makes no Run2 claim.

## Evidence and identities

Each input ROOT payload and JSON manifest is hash-bound; highest-cycle tree counts,
branch dtypes, event identities, raw-time declaration and all-event denominator
are checked. Repeated ROOT payloads cannot be counted as additional samples. The
manifest and payload hashes are rechecked after reading/classification. An existing
output directory is refused. A failed run may retain partial ledgers, but never a
completed `summary.json`.

For each file, the outputs are:

- `inputN_steps.parquet`: original ordered ROOT row, full `(run,subrun,event)`,
  event-row join, bar/sector, bunch phase, raw endpoints, deposited energy, stock
  wrapped endpoints, historical folded endpoints and every classification flag.
- `inputN_frames.parquet`: **all** event rows, including empty or rejected frames,
  with per-flag counts. Its order matches the source event tree.
- `summary.json`: file/manifest/code/geometry/source hashes, row counts, ledger
  hashes, per-sector/per-phase counts (including zero bins), totals and measured
  elapsed times. `F` and `finite_library_uncertainty` are null;
  `timing_closure` is false.

A step's stable key is **input SHA256 + full event tuple + original zero-based
ROOT row**. Repeated tuples across different replay files are not merged. These are
identities within a fixed replay, not independent source showers or photon IDs.

## Timing definitions

Let `w = -float32(pbtime)` and `T = 1695 ns`. The historical fast path uses

```
shift = floor((t0 - w) / T) * T
historical_t0 = t0 - w - shift
historical_t1 = t1 - w - shift
```

**Both endpoints move by the start-time shift.** The audit regression-tests this
against `fastframes.fold_steps`; independently wrapping the second endpoint would
be a different operation and would distort interpolation across a period boundary.

The stock ZS step eligibility predicate wraps each endpoint separately using
corrected `fmod` in absolute coordinates. It rejects both endpoints below
`w + 100 ns`, both above `w + 1500 ns`, or start above the latter and end below the
former. The three rejection reasons are disjoint. NaN endpoints are a separate
skip category; infinities fail closed. Bounds are strict for rejection, so equality
is retained. No interval-overlap approximation replaces this endpoint heuristic.

`historical_only_eligible` means finite endpoints accepted for possible historical
optics but rejected by the stock **step** predicate. It does not mean that the step
actually generated photons, passed SiPM cuts or contributed to a veto. Counts and
summed deposited energy are not response weights.

Additional flags report period-crossing steps, endpoints outside the initial
period, an end outside the historically folded period, negative/multiperiod
durations, eligibility changes if the float32 bunch-time cast is omitted, and
floating-point residuals between historical start folding and stock corrected-fmod.
The latter contrast preserves operation order; it is not a proposed timing fix.

## Rechecking saved evidence

`--verify /data/existing_step_timing_audit --counters ... --stock-source ...
--out /data/new_verification.json` hash-checks the original inputs and saved ledgers,
then reclassifies the saved raw endpoint columns. Every ledger column and per-input
summary must reproduce exactly. It does not reread the ROOT physics trees or rerun
optics. The verification report retains up to 100 exact roundoff rows and frames
without a stock-eligible step for inspection; totals and complete rows remain in
the original ledgers. This also permits checking a prior census against the pinned
geometry guard without replacing its original code/provenance record.

## Remaining optical experiment

The census cannot test arrivals delayed across a boundary by optical propagation,
photon counts, shared RNG, union-of-channel response, full waveform tails, fitted
pulse membership or frozen-grid coverage. A large step rejection count alone
cannot attribute the fast/stock F difference to timing. Preserve these ledgers as
the fixed replay row map for a future common-photon experiment, retaining separate
historical, arrival-wrap, step-rejection and combined arms. Selected-track exposure
and upstream library uncertainty remain separate gates.
