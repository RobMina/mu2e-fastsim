# Proposed native target recording (unapplied diagnostic draft)

This is a source-review proposal, not an installed recorder, an approved simulation or a populated library. An unapplied, uncompiled [native diagnostic draft](native/README.md) now supplies source assets and fail-closed source-assembly checks; it is always library-ineligible and does not implement a launch frontier. The prepared configuration-only diagnostic cannot provide the native fields below. No live Offline module or default is changed.

## Chosen development interface

Start with the union of explicitly enumerated active target-material placements, not target end disks and not a mother-volume shortcut. The exact placement membership and material configuration remain unqualified until a matching native geometry inventory is obtained and reviewed. A target-plus-support envelope is a different interface and must not share the same boundary identifier.

Keep the full event's first outward crossings together, including every species, charge, statistical weight and companion, with zero-exit events explicit. Persist all later crossings in the diagnostic ledger. Do not independently resample particle marginals. Initial representations must not condition on downstream survival or a truth-based acceptance mask.

The transport arm retains the target and its material dynamics for re-entry. Precomputed stock-field returns cannot replace those dynamics if the exterior transport changes. Primary particles and all later external branches remain part of the coupled event. A first-exit source is not by itself a closed Markov interface for arbitrary exterior feedback, nor a replacement for protons that miss the target and later interact elsewhere. Its initial validated scope must distinguish target-origin production from full beam-line event generation.

## Launch frontier and causal ownership

A diagnostic crossing is not automatically a launchable source state. Build a causally ordered event graph with track-continuation edges and daughter-birth edges tied to the actual parent step/production vertex. Starting from each original primary history, admit a first outward transfer only if that point is not already causally downstream of an admitted transfer. Transfer ownership of the continuing track and its future branching to the transport arm; suppress all later stored crossings on that owned continuation and descendants born after transfer. Those rows remain diagnostic observations, not additional launches.

A daughter already born before its parent's transfer is a separate active branch and may itself supply a frontier state; suppressing every genealogical descendant would incorrectly lose that branch. Conversely, a parent that exits, returns and later produces a daughter must not be launched together with that stored daughter's later exit. The transport arm generates the return and subsequent branching. External descendants after transfer are owned by that arm too, including feedback into the target. Earlier exterior branches or primary histories outside this target-only scope need an explicit upstream owner and cannot be silently dropped.

Constructing this frontier requires native parent production-step/vertex incidence and unambiguous ordering relative to transfer, not merely global-time sorting or the final parent pointer. Same-step daughter production at a transfer boundary is unresolved until native callback/secondary semantics establish ownership; equal timestamps are not a tie-break rule. Persist candidate/admitted/suppressed/unresolved classifications with the owning transfer identity and reason. Unresolved ancestry or vertex/transfer ordering prevents library admission. The frontier algorithm and these cases remain unimplemented and require independent validation.

## Default-off recorder contract

The source proposal must use a dedicated opt-in recording path and leave the disabled path unchanged. Review and build the exact source/runtime combination before proposing a run. The recorder must not kill particles, modify weights, change physics cuts, draw random numbers or force navigation. Native volume/status observation is not a request to recompute intersections with a second navigator.

### Native geometry manifest

Persist a stable, file-local placement identifier and complete root-to-leaf touchable/copy history, parent identity, logical-volume/solid/material identity, native placement transforms and target-membership classification. Cover placements that can appear in relevant pre/post touchables, including world escape and non-target neighbours. Names or copy numbers alone are insufficient. Replicas, parameterisations, reflected placements or unsupported hierarchy cases must be explicit and fail closed until their instance identity is supported.

Review target membership against the constructed geometry and configured target materials. Ancestor membership alone is unsafe for vacuum or different-material daughters. For a union transition, classify the actual pre/post physical-region membership; exclude target-to-target internal interfaces. Store native observations separately from any reconstructed surface normal or analytic geometry interpretation. A manifest hash binds every event/crossing record to this geometry inventory.

### Event and exposure ledger

| Field group | Required observations |
|---|---|
| Source identity | Original job/shard, process history, event ID, source config/runtime hashes and geometry-manifest hash |
| Generated source | Generator collection/ProductID, all primary keys/PDGs/states/weights, generator count and original primary links |
| Completion | Begun/completed/aborted outcome, original G4 status, track-limit and field-propagator diagnostics |
| Recording accounting | Attempted and stored crossing/lifecycle rows, independent overflow flags and first offending identity |
| Exposure accounting | Persisted generated-event/proton counts, SubRun count products, requested IDs and complete job termination/log reconciliation |
| Empty case | Zero target crossings only after event, recording and primary-link completeness checks pass |

The native draft now observes `G4Event::IsAborted()` separately from Mu2e's limit-derived `StatusG4` and refuses recorder completion for an aborted event. It publishes only after SD handoff; even then this is recorder end-hook/ledger completion, not enclosing callback return, successful run termination or persisted art. The [source/POT adapter plan](TARGET_SOURCE_ADAPTER.md) separates requested source, native outcomes and written-product reconciliation. Its uncompiled [v2 source schema](TARGET_SOURCE_SCHEMA.md) now proposes actual generator/primary-helper/insertion and in-memory reciprocal-link recording; a separate [uncompiled original-product reader and structural join](TARGET_READBACK.md) now propose written-product/EventID/SubRun checks, but actual original-art verification and qualified exposure remain absent.

The recorder needs an explicit policy for failures before normal event output. A sidecar completion manifest and retained full logs may be necessary; neither is replaced by an end-SubRun count. Never regard an unwritten event as zero. Do not remove capped/bad events from a full-exposure denominator and call the remaining distribution inclusive. Prevent overflow or recover the complete intended shard without selecting the event population; validate any restart procedure and duplicate/overlap accounting independently.

### Crossing ledger

Retain every native transition touching the target-member set, including entering, leaving and internal target-to-target transitions. The raw diagnostic ledger must distinguish these; the derived outward source excludes internal interfaces only after membership is verified.

| Field group | Required observations |
|---|---|
| Identity | Event, SimParticle collection/ProductID/key/stage, original parent pointer, native track/parent IDs and step ordinal |
| Paired endpoints | Pre/post position, momentum, global time, proper time, kinetic energy and statistical weight, each explicitly tied to its endpoint |
| Native geometry | Unique pre/post manifest IDs and touchable/copy histories, with explicit null-world and unavailable cases |
| Native status | Pre/post step statuses, process defining the step, observed track status and common/stepping-cut outcome where applicable |
| Classification | Native target-member booleans for each side; entering/leaving/internal/touching/ambiguous labels and crossing ordinal |
| Source provenance | Runtime/source/config/geometry hashes, coordinate system and units |

Record at the appropriate user callback before the configured common stepping cut, and separately identify the effect of cuts where observable. Source inspection supports the existing stepping-only-before-common-cut order, not a new recorder's runtime correctness. A boundary-step status alone does not prove an outward union crossing; membership, touching/zero-length cases and lifecycle transitions must also be resolved. Do not invent a normal from momentum, count a geometry-tolerance nudge as physical movement or silently classify ambiguous membership as a loss.

### Lifecycle and ancestry ledger

Persist track birth/end observations sufficient to identify particles born inside target material, born outside, created at a boundary, killed by configured cuts, stopped/decayed, suspended/resumed or postponed. Native track numbers alone are not persistent collection identities. Bind them to the original SimParticle and primary pointers without proximity/time matching. Retain daughter relationships, native parent production-step/vertex incidence and same-event companions so the causal launch frontier can distinguish births before and after a transfer; do not promote particle-end states to exact crossings.

Birth inside a target is distinct from an inward crossing. At-rest and in-flight daughters need their actual lifecycle/boundary observations, not an inferred parent terminal time. Delayed descendants, world escape, missing products, unresolvable pointers and unsupported lifecycle states must remain explicit.

## Review and validation sequence

1. Pin the intended source/runtime/configuration and review the native inventory/membership algorithm, callback placement, identities, data products and independent row counters. Check inherited common/stacking cuts and exterior-return scope. Keep all changes isolated and default-off.
2. Have the user build the reviewed implementation in an appropriate isolated area. Verify source/plugin/runtime/library freshness before executing anything against it. This proposal is not build-ready source.
3. Propose a no-transport readback/schema control if supported without native geometry construction; it cannot establish geometry membership or crossing correctness. Native probes, geometry construction and Geant4 transport require their own explicit approval.
4. Only after approval, run one primary event first with hard time/output caps and complete raw products/logs. Validate persisted primary links, event/exposure counters, geometry manifest and row conservation. Stop on ambiguity, missing fields, overflow, unresolved ancestry or bad termination; do not retry blindly.
5. If the first gate passes and the approved scope permits, use a small multi-event diagnostic with an independently reviewed feature-coverage plan: radial/fin exits, target interfaces, both charges/species, births inside/outside, re-entry, world escape, cut termination and zero-exit events. A small random sample may not contain these cases. Missing cases stay unvalidated; no physical population filtering or forced-case result is a yield measurement.
6. Test persistence and fail-closed admission, including injected overflow/missing-field/duplicate-ID controls that do not claim physical coverage. Reconcile original event IDs and all raw links, not only crossing-row counts.
7. Estimate production size/cost from completed diagnostic logs, not a linear timing extrapolation or downstream benchmark. Request production statistics separately, with disjoint development/held-out source shards and paired errors. No populated library is certified by the diagnostic alone.

## Resource and authorization boundary

The previously drafted 100-primary-event configuration is only a recording-study placeholder. It is not the native implementation above and must not be submitted as one. Its stock timing basis excludes extra observer I/O and is not a prediction of diagnostic runtime or memory. A future request must state the exact events, site/runtime, wall/CPU reservation, output cap, stop conditions, recording gates and recovery policy. Source preparation, a user-run build, native checks and Geant4 execution are distinct approvals. No campaign, grid submission or physics-validity claim follows from this proposal.
