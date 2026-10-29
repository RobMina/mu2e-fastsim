# Saved-refit frozen-grid coverage

`veto.refit_saved_coverage` adds per-frame coverage to the already verified saved
ADC-refit coincidence products. It does not rerun the coincidence finder, any
upstream response or fitting, or an earlier verifier. Successful generation and
verification modules and artifacts remain immutable.

## Evidence and admission

The saved-refit summary, original completion receipt and saved-output verification
report require external SHA256 pins. A summary/report pin alone does not bind the
native artifact bytes: a modified cluster file and resealed receipt could otherwise
retain those pins. The separately pinned original receipt closes that gap.
Require exact schema, scope, full-registry identity,
completed structural-verification gates and matching source snapshots. Hash the
complete refit transaction against its expected inventory and original writer's
source ordering, without decoding pulses, waveforms, calibration or native members.
The native bridge and member checks are inherited evidence, not re-executed checks.
Floating cluster properties remain hash-bound finder observations.

Follow the pinned verification's source-report bindings to the completed pilot
membership transaction and its previous verification. Require the frozen-grid
verification gate, source map and snapshots, and exact transaction inventory.
Compare every pilot artifact and marker digest with the original transaction
bindings embedded in the pinned refit summary; do not trust a resealed current
receipt as the provenance of the previously verified grid.
Decode its full 9,000-frame registry and existing `grid.parquet`. Compare the full
refit registry exactly, then admit every saved frame and cluster by its full
input-qualified identity, immutable frame ID, block order and count tables.
No subset registry is constructed or reranked. Cluster indices retain canonical
saved ordering; the coverage helper must not conceal malformed saved order.

The existing paired/reference report chain binds the card and selected-track
provenance. The selected-track file is **not opened, hashed or re-histogrammed**.
Its provenance and the grid's previous verification are inherited; this check
makes no assertion about the file's current filesystem state. Validate the saved
4,042-bin grid, contiguous edge bits, 0.25 ns spacing, normalization and historical
metadata: 1,929,421 selected tracks out of 1,930,163, with strict `85 < p_mid < 200`.
The unchanged event-gate card is recovered from the bound reference JSON.

## Coverage semantics

Use the frozen `pilot_coverage.evaluate` implementation, including its independent
integer-cell occupancy-mask parity check. Preserve the historical rule:

```
ceil((coincidence_time + dt - edges[0]) / step)
```

Clip to the grid domain and occupy half-open integer slices. Do not reinterpret
this as a continuous strict window at cell centres.

Each full frame receives:

- `track_grid_coverage`: the official unchanged card result;
- `gate_pass` and `event_gate_override`: the event-gate decision;
- `n_coincidences` and `n_veto_eligible`: separate coincidence counts;
- `eligible_window_coverage`: a **diagnostic** with only the event gate disabled,
  retaining the original coincidence eligibility, time windows, grid and weights.

The diagnostic also passes the frozen helper's integer-mask parity check. Verify
that official coverage equals diagnostic window coverage when the event gate
passes, and equals one otherwise. A gate-rejected frame can therefore have
coverage one without its windows occupying the entire grid. An empty frame remains
present, passes the event gate and has zero coverage. There is no selected-frame
mean, error bar or estimator.

## Bounds and publication

Require at most four frames per input block, 2,048 clusters per block and 20,000
grid rows before decoding; this fixed retained grid must have exactly 4,042 rows.
The full registry has exactly 9,000 entries. Flat Parquet schemas, numeric storage
widths and finite values are checked before use. Input files are at most 32 MiB;
Parquet declared uncompressed data are capped at 128 MiB, with at most 64 columns
in these schemas and 4,096 row groups. Complete transactions remain under 32 MiB.
Coverage allocations use the frozen `max_cells=80000` bound. These are admission
and allocation bounds, not whole-process RSS guarantees.

Recheck consumed evidence and completed transactions before coverage and again
before publication. Executed source snapshots must hash to their declared hashes.
Write a new exclusive JSON report, at most 2 MiB including source snapshots,
outside input and source locations. This is not the atomic-completion protocol
of `BatchArtifacts` and makes no crash-durability claim. Never overwrite a report.

```bash
python3 -m crvfastsim.veto.refit_saved_coverage /data/EXISTING_REFIT \
  --verified /data/EXISTING_REFIT_VERIFIED.json \
  --expected-summary-sha256 REFIT_SUMMARY_SHA \
  --expected-verified-sha256 REFIT_VERIFIED_SHA \
  --expected-complete-sha256 ORIGINAL_REFIT_COMPLETE_SHA \
  --out /data/NEW_COVERAGE.json
```

## Completed retained check

All 208 new targeted tests pass locally in 18.69 s and remotely in 90.32 s. They
use real Parquet and transaction operations with manually asserted synthetic
authority, not detector physics validation. Forbidden-entrypoint guards cover
physics, upstream admission, prior verifiers and selected-track reads/hashes.
Tests include original receipt pinning, inherited grid resealing rejection, empty
frames, integer boundaries, event-gate decomposition, metadata bounds, late
mutations and exclusive output.

The retained saved-only calculation completes for all three frames, with three
coincidences and 17 native links. Both historical integer-grid parity checks and
the event-gate decomposition pass. All three frames fail the unchanged event gate;
coverage one is therefore an event-gate override, not full window occupancy.

| Full-registry frame | run:subrun:event | Eligible coincidences | Card coverage | Window-only diagnostic | Event-gate override |
|---:|---|---:|---:|---:|---|
| 2504 | 1430:0:2505 | 1 | 1 | 0.0041084864319399445 | yes |
| 3280 | 1430:0:281 | 1 | 1 | 0.0037394638080543332 | yes |
| 5754 | 1430:0:2755 | 1 | 1 | 0 | yes |

The report is 786,465 bytes including source snapshots, below the 2 MiB cap.
Whole-process execution took 7.85 s, with 220,256 KiB peak RSS; the internal
pre-publication observation was 5.532 s. Original refit summary, completion receipt
and verification-report hashes are unchanged. Native structural checks are
inherited rather than rerun; neither selected tracks nor prior verifiers were
opened/executed. No estimator is formed from these deliberately selected frames.

An initial invocation was rejected before admission because its output was inside
the prior verification's protected parent directory. It produced no report. The
successful invocation used a separate sibling data directory; no code or physics
setting changed and the output-exclusion guard was not weakened.

## Scope

These three deliberately selected frames are not representative exposure.
`F=null` and `finite_library_uncertainty=null`; C4 stays stock. Neither per-frame
coverage nor the inherited nonempty structural checks establish same-pulse stock
parity, floating cluster-physics closure, the joint selected-track PBI/time law,
physical library reuse weights, finite-library covariance, timing closure or
full-campaign admission. No historical stock-native comparison is performed.
