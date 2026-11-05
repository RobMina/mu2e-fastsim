# First hybrid arm: readiness and owner handoff

The next R7 deliverable is a **tracker-only substitution with stock calorimeter digis**, not a threshold-only
emulator. CE complete-history stock control and frozen independent confirmation already pass all 52 paths.
A fast result must carry its own complete filter-call history and must not inherit stock decisions or ordinals.

## Required owner response

R2 should identify an existing, already-built injection runtime and provide either a retained combined art
artifact or its exact executable configuration, plus a receipt containing:

- Source shard identities, immutable input hashes and complete source traversal order; exact number of events,
  including no-handoff/no-straw/zero-digi events. Missing table rows must fail, not become invented zeros.
- Original stock Straw digis/ADCs, stock Calo digis and original EWM/PBT, plus replacement Straw digis/ADCs under
  a different explicit label:instance:process. Preserve same-event reference TriggerResults where available.
- Resolved geometry, field, conditions and release/environment; digitizer configuration/seed and which particle
  population is modelled. A primary-only tracker arm must remain labelled primary-only, not full detector response.
- Importer plugin path/hash and verified build/backing. The user builds; a source file or successful Python test
  is not evidence of a compiled compatible importer. No new build is implicitly requested by this handoff.
- Product presence/registry and independently reconciled full event IDs/count/order. All unmodified timing and
  stock-detector serialized products must remain byte-identical through injection.

R6 need only identify the retained full-event input/hash and its stock calorimeter/timing context for this arm.
Fast calorimeter simulation is a later arm, not a prerequisite for tracker-only substitution. A shower library,
response-only numeric table or reco ntuple is not itself a CaloDigiCollection.

## Existing candidate and explicit limits

A production-matched first choice is the complete CE development parent's existing dts population plus its
already validated full stock digi history. R2 can transport that exact source population, preserve zero-output
events and supply replacement straw digis/ADCs; stock calo/timing/reference decisions remain in the original art.
This needs no new Geant4, but a compatible built injection path and explicit source/seed receipts are still required.
Private locations and immutable input hashes are recorded in the operational handoff, not this public document.

The existing unfiltered 2,000-event CE pilot is the smallest bounded full-event candidate. It retains no-response
events and both detector contexts, but was made with MDC2025ay / Offline 13_39, not the CE ba / Offline 13_41
production reference. Its newer exporter does not change its simulation provenance. Use it only as a separately
labelled historical-era hybrid control unless geometry/conditions/runtime equivalence is explicitly established.
It does not supply full production/POT exposure. Selected CE dts references are a different population again.

An existing `StrawDigisFromTable` module can ingest ordered StrawDigi and aligned ADC waveform rows **inside each
original art event**. It checks missing/duplicate identities, integer bounds, flags, repeated digis and ADC lengths.
Such a separately validated keyed table importer is not file concatenation: the framework still reads one original
event and the producer inserts its replacement collection. Full population/zero-event and provenance checks are
required. Its existing no-field/extracted reco FCL is not the Run 1A trigger FCL and must not be reused unchanged.
Original StrawDigiMC associations are not replacement-digi truth and cannot certify fast efficiency.

The inherited step injector emits empty calorimeter/VD products and its example config applies extracted-v04
geometry and PrimaryFilter selection. Those settings fail this contract for Run 1A stock-calo triggering.
Do not retag triggers to empty placeholders or use selected output to hide missing upstream events.

## Current adapter route

An existing user-built p109 importer/dumper bundle has now round-tripped all 2,811 events of the first CE development
parent inside the released ba runtime. The C++ comparison checked every original StrawDigi field and ADC sample;
all six original stock products and all 56 original status words were preserved. This establishes that bounded
adapter, not arbitrary plugin compatibility or fast physics. The read-only source truth export also completed.
The all-52-path unchanged-digi trigger control also passes on that actual combined artifact: 146,172 event-path
comparisons, zero accepted/state/wasrun/mapped-endpoint differences. Raw shortened-path indices remain different.
The first actual primary-only CE transport plus released stock digitization and all-path trigger diagnostic has
also completed on those 2,811 events. It measures 8,790 changed event-path decisions across 52 paths, zero unrun or
exceptional paths; the nine calorimeter-only paths have zero changes. Representative tracker-only CE paths are
about 0.8--0.9 percentage points lower and calorimeter-seeded CE track paths about 5.8--5.9 points lower. This is
not equivalence or a correction: the arm lacks secondaries/returns, retains the stock history's calo response and
uses a separate electronics stream. At that one-parent stage, full original-parent history remained open; the completed development
extension is recorded below. Independent paired fast confirmation remains open.

R7's orchestration uses distinct fast-step and fast-digi labels plus the released stock tracker digitizer. Original
stock calo and timing remain in each event. The fast electronics use an explicitly recorded seed/engine stream,
not an assertion of the original production random draws: observed migrations can include response reseeding as
well as changed steps. All original source events, including no-handoff/zero-step cases, must survive injection.
Initial one-parent testing was not a full-production-history control; the development extension below
subsequently ran all parents in one trigger process.

## Complete-history development result

The first-parent result above has now been extended to all ten original parents: 28,044 stored events,
740 explicit zero-step events and all 52 trigger paths in one continuous trigger history. Representative
TPR CE acceptance shifts -1.24091 percentage points and CPR CE -5.80516; all nine calorimeter-only paths
remain unchanged. The deficit is not physics closure, and common-seed parent resets do not supply an
independent-parent SEM. Independent fast confirmation remains open.

A separate measured-feature export preserves exact original/replay decisions against that hybrid baseline.
The two BO-final selections have zero eager/JIT measured track/count/filter/composed-bit differences, retaining
actual upstream fast reach rather than emulating it. The quiet pair measures no whole-job acceleration;
a separate CPU final-selection microbenchmark cannot replace dominant reconstruction. See
[JAX scope](JAX_SELECTION.md) and [response-control status](RESPONSE_CONTROL.md). The latter preserves
importer/particle-state and electronics-history uncertainties: only its first parent has completed, not the
full-history C trigger. R7 is paused with no jobs running; the remaining nine parents are unlaunched. See
[the pause/resume gates](PAUSE_RESUME.md) before treating the work list below as executable.

## R7 work after admission

1. Establish a matching all-path stock control for the actual admitted input/runtime, reusing finished references.
2. Confirm injected identity coverage and byte preservation of stock calo/timing/original reference decisions.
3. Retag every enabled straw consumer, including direct DigiFilter instances; leave trigger cuts, timing and
   source population unchanged. Reconstruct calorimeter products from the same retained stock calo digis.
4. Run all 52 paths over complete history in one process, evolving fast-arm reach counters independently.
5. Report per-path pass counts, both migration directions and paired differences. No assertion of exact fast/stock
   equality is made in advance. Measure uncertainty across independent paired shards, not two file prefixes.
6. Only after validity, perform quiet like-for-like timing; prioritize shared pattern recognition/fits for JAX.
