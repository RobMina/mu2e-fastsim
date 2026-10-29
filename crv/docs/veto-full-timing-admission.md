# Full retained timing campaign: remaining admission contract

## Scope and decision

The next timing campaign would evaluate the **same 9,000 retained frames**, four
frozen timing arms and two paired seed domains against the **existing fixed-stock
outputs**. This is a conditional historical-mixture comparison, not a corrected
CE-conditioned sample or finite-library uncertainty measurement. No stock rerun,
new mixture, build or submission is implied by this design.

**Execution is not yet admitted.** The bounded pilot cannot simply be enlarged.
Its physics kernels are reusable, but campaign storage, complete receipts and
aggregation need a separate versioned implementation. Prior successful controls
must not be rerun merely to reproduce their evidence.

## Reuse rather than repeat

- Frozen 9,000-frame source-qualified registry and 3,072-batch retained plan.
- Completed full-input indexed-cache audit and original row fingerprints.
- Saved keyed optical smoke, all-live response, nine-frame streaming pilot,
  saved membership, nonempty refit, frozen-grid coverage and stock/refit evidence.
- Original raw mixtures, condition assets, calibration, lookup/deviation/template
  files, selected-track provenance and stock products.
- Frozen 4,042-bin, 0.25 ns coverage grid: 1,929,421 selected tracks of 1,930,163,
  strict `85 < p_mid < 200`, with the existing ceil-grid and event-gate semantics.

Admission must bind original completion receipts as well as summaries and
verification reports. A resealed child report alone is insufficient. Consume
saved evidence directly; do not invoke old generation or verifier entrypoints
that unnecessarily reread raw or stock products.

## Minimal new implementation

1. **Full schedule and import ledger.** Exactly one disposition per original
   frame/batch/seed: imported from a compatible original receipt or newly executed.
   The nine pilot frames form complete planned batches and can be imported.
   Smoke frames have a different grouping and need an explicit provenance-preserving
   mapping or residual schedule. Older extreme controls have different draw/domain
   contracts and cannot silently become campaign children.
2. **Hierarchical campaign receipts.** The pilot's flat top-level child list
   exceeds `BatchArtifacts`' 128-artifact limit at full scale. Add a bounded index
   hierarchy and a completion verifier, not an increased constant alone.
3. **Exact bounded global optical-root uniqueness.** `streaming_optics.RootUnion`
   retains and repeatedly sorts up to 1.8 million roots in memory. Replace its
   campaign use with an exact external/bounded design, validating collisions across
   inputs, batches and seeds before final acceptance. Hash summaries alone do not
   prove uniqueness. Preserve full source keys and existing per-batch validation.
4. **Explicit persistence and total-space budget.** Cover permanent products,
   root-index scratch, temporary writes, logs, filesystem overhead and incomplete
   attempts. An exceeded cap leaves the campaign incomplete; it never drops a
   frame, tail, channel, arm or seed from the denominator.
5. **Streaming native membership and coverage.** Reuse tested adapters and native
   pulse/time bridges, but replace nine-frame accumulation with bounded children.
   Preserve ignored-channel policy, all sectors/ends and all-live thermal domains.
6. **Full-stage comparisons and paired F consumer.** Include candidate arrival and
   boundary classifications, zero suppression, complete waveform-tail comparisons,
   pulses, ordered native members, event-gate flips and frozen-grid coverage.
   Pair by source-qualified original frame; average response replicas within that
   frame before estimating paired uncertainty.

## Storage is a real blocker, not a runtime projection

The current pilot permits 64 MiB per batch/seed child,
including four response arms. For 3,072 batches and two seeds:

| Per-child cap | Aggregate child cap |
|---|---:|
| 64 MiB | 384 GiB |
| 8 MiB | 48 GiB |
| 4 MiB | 24 GiB |
| 3 MiB | 18 GiB |

These are **cap arithmetic, not predicted output sizes**. Metadata, scratch and
failure reserves are additional. Four MiB per child has inadequate headroom;
three MiB fitting arithmetically does not show that real children will fit.

Choose either a separately approved larger storage destination or an explicitly
reviewed compact persistence schema. A compact schema should avoid repeated raw
inputs/calibration while retaining candidate photon identities (including rejected
candidates), frame/arm/seed accounting, pulses and native bridges, membership,
coverage, stage-difference evidence and provenance. Omitting intermediate arrays
reduces what saved-only verification can independently reconstruct; state that
limitation rather than claim equivalence with full-stage persistence. No existing
evidence is to be deleted to make space without approval.

## Validation before any broad execution

First test the new schedule/import/persistence/aggregation implementation using
already saved pilot products, with no optics, response, fitting or completed
finder rerun. Exercise malformed receipts, missing/duplicate frames, late source
changes, collisions, quota/cap failures and interrupted publication.

Only if that implementation introduces an unmeasured resource property should a
new bounded execution be proposed: predeclared untouched whole frames chosen from
workload descriptors, targeting that property. It is not another generic closure
control or a representative F sample. Never extrapolate runtime from small-N;
record actual progress and resource measurements.

Completion requires all **9,000 unique original frames** in every arm/seed,
including empty frames: 72,000 frame/seed/arm ledger rows, not 72,000 independent
exposures. Full receipts and the unchanged stock pairing must pass before F is
published. Scientific acceptance criteria remain separate from execution success.

## Statistical limitations remain explicit

The nine-frame pilot has zero coincidences; it is not evidence for zero full-sample
F. Three nonempty refit frames have card coverage one because of the event gate;
their tiny window-only coverages do not estimate a sample mean. Exact unordered
member sets there do not prove all-input native-finder equivalence. Two coupled
seeds provide descriptive seed contrasts, not a robust response-variance estimate.
Physical source reuse, cross-species ancestry and the CE selected-track joint
exposure law are separate open gates. No independent-file bootstrap or count-ratio
normalization supplies them. C4 remains stock until the relevant gates close.
