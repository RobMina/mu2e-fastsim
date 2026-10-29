# Saved stock-ADC-refit finder diagnostic

`veto.refit_membership` is a separately versioned, bounded finder-only check of
already-saved ADC-refit pulses. It never refits ADCs, draws photons or response,
reads raw/stock ROOT files, reruns prior verifiers, or computes coverage or F.
The previous empty streaming pilot and legacy nonempty coverage controls remain
unchanged. This runner admits both retained blocks before either finder call.

## Inputs and authority

Require external pins for the original downstream summary, boundary summary,
final downstream verification, completed membership summary and its saved
verification. Reuse the membership transaction's complete 9,000-row registry,
its hash-bound 15,360-row per-input calibration, the exact Run1A counter geometry,
and the pinned pulse-finder source plus all three `ignoreChannels:true` runtime
configurations. Check the completed transaction inventory and code snapshots.
No subset is refrozen, reranked or assigned invented frame IDs. This admission
inherits the named prior raw/source verification rather than repeating it.
The local boundary inventory is not a remote input and need not be uploaded.

The fixed retained targets are block0/event2505 and block1/events281/2755, qualified
by their original input SHA and the full saved run/subrun/event identity. Require
all three complete frames, 87 refit pulses (25 + 62) and 80 saved waveforms. Empty
finder results remain valid observations; there is no seed retry or cut tuning.

Read only the saved event census, stock pulse scalars, refit pulse/waveform/
comparison tables and conditions. Parquet admission precedes decoding: 32 MiB
file, 128 MiB declared uncompressed, 64 columns, 4,096 row groups; row limits are
9,000 registry, 15,360 calibration, 20,480 conditions and 2,048 pulse/waveform/
comparison rows. These limits do not certify process RSS.

## Pulse bridge and time contract

Restore absent run/subrun only from the bound complete stock pulse identity
collection. Match the source-qualified tuple to the immutable full registry.
Check saved bunch time against registry rounding and saved raw bunch time when
present. Compare every supplied condition against complete authoritative channel
calibration, including offset, status, pedestal, pulse-area/height and ROC/FEB.

Recover start TDC only through the genuine saved `(wf,event,chan)` link. Require
stored pulse order to equal native `(wf,seq)` and structural
`(frame_id,chan,starttdc,seq)` order before calling the adapter, which itself sorts.
Retain original fit flags. Check both original relative time (peak plus offset)
and reco time (relative minus float32 bunch time) bit-exactly; never reconstruct
original relative time by an inverse bunch shift.

The original comparison table maps stock collection index through actual stock
digi references to `(wf,seq)`. Its older retained schema omits all three channel
fields (`bar,sipm,chan`): recover them only by a one-to-one full-event/stock-index
join to the bound stock pulse collection, then cross-check the actual waveform.
A partially present channel schema fails instead of being repaired.
Require complete stock/comparison/active-refit
bijections, exact copied scalar values and exact saved PE/time deltas. An unmatched
outer-join row fails this bounded contract rather than disappearing in an inner
join. Stock indices and refit indices remain distinct. Refit values are **not**
identical stock values, and the historical full-length fit is not relabeled as
the newer width-bucketed diagnostic fitter.

## Execution and output

Use the frozen `pilot_coincidence.run` once per input: at most two calls, unchanged
four-frame/2,048-total-pulse/512-per-frame/two-million candidate-tuple limits.
This fixed candidate must remain all-active: an ignored row now fails admission
rather than being dropped or silently changing the saved comparison's domain.
The frozen adapter's broader policy still retains diagnostic ignored rows and
excludes them only from finder input. Nonlive status bits also fail admission.
No fit-quality cut is added.
The finder receives saved original relative times and integrated float64 PE.
Restore all real identities on native outputs and verify the resulting pulse
index map against the admitted map. Save original pulse fields/flags separately
from the adapter's pulse collection, plus waveforms and stock-to-refit bridge.

One exclusive atomic transaction, capped at 32 MiB including receipts and source
snapshots, records registry, calibration, complete frame census, pulses, clusters,
ordered native members and bridge. Consumed inputs and code are rehashed before
dispatch and before completion. A failure after dispatch leaves an incomplete or
failed transaction, never a successful certificate. The transaction seed label 0
is protocol-only metadata for this deterministic saved-refit check; no new draw or
exposure is represented. See `veto-batch-artifacts.md` for durability limitations.

No stock native-cluster comparison is made here: the historical boundary export
lacks some ordinals and full scalar properties. Even a later exact mapped-member
comparison would test robustness under changed pulse values, not same-pulse
stock finder equivalence. Nonempty output can exercise the real adapter path but
does not establish timing closure, efficiency, representative exposure, finite-
library uncertainty or promotion. `F=null`; C4 stays stock.

## Completed bounded observation

The retained three-event check completed with two finder calls. Full registry
frames 2504, 3280 and 5754 contain 25, 36 and 26 saved pulses respectively; each
produces one coincidence, with 8, 4 and 5 native members. All 87 pulses and the
80 genuine waveform links were preserved; no ignored/nonlive pulse was admitted.
This exercises real nonempty adapter output, not a representative sample.

All 87 refit PE values differ from their corresponding stock values; 64 reco
times differ. Maximum absolute differences are 2.9390826767894396e-6 PE and
1.0692929208744317e-8 ns. Failed/zero-NDF flag comparisons agree. These nonzero
numeric differences remain explicit, never promoted to exactness by tolerance.

The output occupies 1,109,847 logical pathname bytes under the 32 MiB cap.
Measured whole-process runtime is 9.97 s with 237,912 KiB peak RSS. The 231 new
targeted tests pass locally and remotely; fixtures are synthetic and the runner
tests stub finder dispatch. The separate saved-output structural verifier
(`veto-refit-verification.md`) now passes without finder regeneration, with 124
additional targeted tests passing locally/remotely. It verifies the bridge,
native structure and complete frame census, not independent scalar physics.
Nested inventory diagnostics describe pre-dispatch checks only; actual
finder observations are in the final frame/count tables and the top-level
`nonempty_real_refit_membership` flag. Stock-native comparison, coverage, timing
closure, exposure and finite-library uncertainty remain separate open work.
