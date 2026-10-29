# Saved-pulse coincidence and frozen-grid pilot

The separate `veto.pilot_membership` diagnostic consumes the completed bounded
streaming pilot. It never regenerates photons, avalanches, waveforms, fitted
pulses or stock reconstruction. It produces no F estimate or uncertainty.
The bounded retained-pilot execution is complete. Its coincidence collections
are empty in every arm/seed: this validates the real empty-output/frame path, not
nonempty real membership, coincidence efficiency, timing closure or stock parity.

## Admission and immutable evidence

Require external SHA256 pins for the pilot summary, its completed saved verification
report, and the historical paired report binding the selected-track Parquet and
reference card/grid. Admission reuses that specific successful prior verification;
it **does not repeat** upstream raw ROOT admission. It freshly validates every
pilot top/child transaction, exact artifact inventories, copied receipts, saved
registry, frame/seed/arm completeness, counts and generation source bindings.
The prior verifier source must match its pinned report. Expected artifact hashes
come from the validated receipts, never a new snapshot of potentially changed
files. All transactions and bound inputs are checked again before publishing.
This explicitly differs from rerunning the upstream verifier, whose inherited
admission reads raw/stock ROOT files.

Output is exclusive and outside all input evidence directories, with a fixed
32 MiB logical pathname-byte cap including receipts/failure reserves. Existing
`batch_artifacts` atomic publication and durability limitations apply. Failures
never truncate rows, renormalize exposure or create a complete transaction.
Input metadata, named allocation ceilings and output bytes are not RSS guarantees.

## Native pulse bridge

`pilot_coincidence.run` uses the unchanged Run1A reference finder with opposite-side
attachment enabled, secondary pulses retained and integrated PE in float64.
The saved fitted-pulse collection must agree with structural order
`(frame_id, chan, starttdc, seq)` and its `(wf,seq)` order; the runner rejects
rather than silently repairs an inconsistent native ordering. Waveform labels
are local metadata, not cross-arm physical identities. Failed-fit and zero-NDF
flags are retained as provenance; no new fit-quality cut is applied.
Saved diagnostic fits also retain status-bit-1 ignored-channel pulses. The reviewed
fixed stock `ignoreChannels: true` policy is bound to all three effective runtime
configurations and the stock pulse-finder source. All saved rows and original
indices remain in the output; ignored rows are explicitly excluded only from the
finder input. Native member links may therefore contain gaps in original pulse
indices, but must never reference an ignored pulse. Counts separately report
saved, ignored and finder pulses. This is the stock channel policy, not an
unrecorded removal or a fit-quality selection.

Each input is handled separately, with the complete frozen registry retained.
Temporary helper labels use immutable frame IDs; every saved output restores
`(input_sha256,run,subrun,event,frame_id)`. Neither packed native triples nor a
reranked subset can collapse repeated event triples across different inputs.
Empty frames remain explicit.

Saved `time_reco` must exactly equal `time_relative - float32(pbtime)`.
The finder receives the saved corrected `time_relative` directly: undoing the
bunch-time addition numerically could otherwise introduce a round-trip residual.
Cluster times are restored to the reco frame by the unchanged recovery helper.
No additional channel-time correction or PE float32 conversion is made.

Limits can only decrease: four frames, 2,048 pulses total, 512 per frame, and a
conservative two-million candidate-combination ceiling. For each frame, sector
type and readout side, products of distinct-layer multiplicities upper-bound
all combinations allowed by the type's coincidence-layer requirements. Counts
precede PE/time filtering and clustering, so this remains conservative without
counting impossible same-layer or wrong-size combinations. This bounds candidate
tuples, not every operation in the finder. Ledger pulse counts are admitted
before decoding; geometry-aware combination counts are checked after bounded
pulse decoding and before finder dispatch. The only per-event dispatch is inside the existing reference
recovery API, using flat NumPy views. Joins and membership/coverage analysis are
vectorized.

Native links are checked against the exact pulse collection, frame identity,
sector type and layers. Saved tables include every scalar coincidence property,
ordered membership links and the pulse index/structural-key bridge. Cross-arm
comparisons use exact structural identities and numerical equality, never
nearest-time matches or a tolerance. Matching cluster indices by order is only
a structural comparison; changed indices are not inferred physical matches.
There is **no new same-pulse stock-finder control** for these pilot pulses.

## Frozen-track grid, not an estimator

`pilot_coverage.frozen_grid` requires the frozen card SHA, strict reference momentum
window, selected-track count, and original grid spacing. File, row-group, numeric
jagged-leaf and uncompressed-byte metadata gates precede track-column decoding;
track-count and bin limits precede histogram allocation. Only `t_mid` and `p_mid`
are read. No track refit, ntuple read or replacement selection occurs.

`evaluate` preserves all frames, builds jagged coincidences from explicit counts,
and records coincidence count, eligible count, gate pass and per-frame weighted
coverage. It calls the unchanged historical `deadtime.frame_coverage` in bounded
blocks and checks exact parity against an independent integer-bin occupancy mask.
The historical rule uses `ceil((t + dt - edges[0])/step)` and clipped half-open
integer slices. It is **not** silently reinterpreted as an exact continuous strict
window at grid centres. Boundary regressions expose that distinction.

Event-gate failures have coverage one, separately recorded from coincidence
eligibility. Zero-coincidence frames pass the gate and have zero coverage.
The complete pilot should yield 72 frame×paired-seed×arm rows but still only nine
unique frames. No mean over that selected pilot, error bar, finite-library
uncertainty, CE factorization claim or full-campaign admission is produced.

```bash
python3 -m crvfastsim.veto.pilot_membership \
  --pilot /data/completed_pilot --verified /data/completed_verification/verified.json \
  --paired /data/historical_candidate/paired.json \
  --expected-pilot-sha256 PILOT_SHA --expected-verified-sha256 VERIFIED_SHA \
  --expected-paired-sha256 PAIRED_SHA --out /data/NEW_MEMBERSHIP
```

The parent of the new output must exist. This command has no remote launcher,
resume/overwrite mode, grid submission, response regeneration or estimator path.
`execution_admitted=false`, `timing_closure=false`, `F=null` and
`finite_library_uncertainty=null` remain mandatory. C4 stays stock.

## Saved-output verification

`veto.pilot_membership_report.verify(directory, out, expected_summary_sha256=...)`
requires an external summary pin and validates the exact saved transaction,
source snapshots, inherited pilot receipts, pulse bridge, native link structure,
full frame/arm/seed census and frozen grid. It recomputes per-frame coverage and
cross-arm comparisons from saved products, **without calling the coincidence
finder or regenerating any upstream physics**. Coincidence scalar values remain
hash-bound generation observations rather than independently reconstructed physics.
Text storage normalization accepts validated string/large-string representations
without changing any string values; integer/float/bool schemas remain strict.

The report requires a new file outside every input evidence directory and repeats
all source/output transaction checks at the end. `--compact` exposes explicitly
whether any nonempty membership or nonzero coverage was exercised. Empty real
coincidence output cannot certify nonempty behavior; nonzero synthetic integration
and mutation tests remain separate evidence.

```bash
bash scripts/verify_veto_pilot_membership.sh \
  /data/EXISTING_MEMBERSHIP SUMMARY_SHA /data/NEW_VERIFICATION
```

This driver runs the new verifier tests and snapshots the current source files.
Previously completed adapter, coverage and runner regression logs remain separate;
it neither claims a full test-suite run nor reruns any saved real finder products.
