# Production-target library acceptance contract (experimental)

No inclusive target-exit library is supplied or certified by this package. A catalogue name, target energy-deposition histogram, downstream-surviving ancestry table or target-end disk is insufficient. This contract separates a usable input library from diagnostic records and applies before any yield table or sampler is promoted.

## Boundary and factorization

Choose explicitly between the union of active target-material surfaces and a closed target-plus-support envelope. The envelope includes transport through its interior vacuum and support material; it is not the bare target-material surface. Pin geometry, materials, placements, primary beam, field configuration, physics list, range/production cuts and native tolerances. A fixed library is conditional on that configuration, not a surrogate for arbitrary target designs.

Account for all relevant outward directions, including radial and fin exits, not just the target endcaps or downstream acceptance. Keep both species charges, pions, muons and the proton/kaon/neutral population needed by the factorization. An acceptance truncation requires a separately justified conservative acceptance support; observing only survivors cannot establish it.

Retain every recorded crossing and its ordinal, direction, event and particle ancestry. First outward crossing is a defined candidate interface, not automatic permission to ignore later target re-entry. Before using a first-exit library, either transport re-entrants through the retained target region or establish a domain where return is excluded. Launch states must form a causal source frontier: a transferred track owns its subsequent continuation and branches, so later stored exits on those histories are diagnostic, not additional launches. Daughters born before transfer can be separate active branches; parent identity alone cannot distinguish these cases. Native production-step/vertex incidence and unambiguous transfer ordering are required, with unresolved same-step cases refused. Independent samples of one-particle marginal histograms do not preserve an event's multiplicity, companions or decay correlations.

## Minimal persisted evidence

- A record for every generated primary event, including zero-exit events, status and statistical weight.
- Independent generated/processed exposure, original job/shard identity, successful completion and resource-limit/truncation diagnostics. Do not count selected events or sum overlapping parent and concatenated products as POT.
- Full event, collection/ProductID, simulation-stage and particle/parent identities; creation process and birth state where required for ancestry.
- Boundary identity and orientation, pre/post position and momentum, global/proper times with explicit endpoint semantics, crossing order and weights. Never combine a post position with an undocumented pre-step time/momentum as an exact boundary state.
- Immutable source, configuration and runtime receipts. Retired catalogue records are discoverable with the explicit read-only `tscensus --include-retired` option; they do not prove that a replica exists.

## Released recording options and limitations

The stock production-target virtual detectors are end disks and are not a closed target boundary. Target-material sensitive-detector collections can retain internal steps and radial exits, but coverage of all relevant target/support solids and native boundary classification must be checked. `StepPointMC` fields require collection-specific interpretation; a copy number is not a unique physical-volume identifier in general.

A configuration-only diagnostic can compose an `intersection` of a volume predicate with a final `constant` false observer that has `write` enabled. The final observer records a step while the enclosing cut returns false, avoiding a new track kill. This must be stepping-only, and the volume predicate uses the track's current deepest physical volume, not recursive mother membership. Actual callback/volume timing, supported released parser behavior and completeness need validation before calling this a boundary recorder.

For an initially small diagnostic, recording a bounded local envelope's steps for all species plus all original event/particle products is safer than manufacturing an exit table from particle end states. Retain original target-material SD steps too. No new simulation is implied or authorized by this design note. A direct native boundary recorder may be necessary if released products cannot resolve post-step time, pre/post physical volumes or daughter/re-entry semantics.

## Prepared diagnostic (not executed simulation)

`fcl/target_library_diagnostic_run1a.fcl` is a proposed 100-primary-event diagnostic derived from the stock POT configuration. Its active path has no event-selection filter, and its output retains raw generator, event-counter and G4 products for successfully written events. It enables all eight target-material SD collections and adds a stepping-only constant-false all-step observer. Existing common physics cuts remain in force. The enlarged observer collection cap is still finite and is separate from the generic sensitive-detector step limit. Any overflow, bad status or incomplete event fails the library gate. It is intentionally not a production campaign or an exit-yield sampler; requested events are not a completed-POT receipt.

The observer records all simulated steps, not just target exits, so this configuration is suitable only for a separately approved bounded diagnostic. Expanded configuration must be inspected under the intended historical runtime. Native boundary status, post-step time and unique pre/post physical-volume identities are not supplied by this observer; no parser or software test can fill those gaps. Check target SD coverage, callback semantics and native exit extraction before requesting larger statistics. Reserved event IDs and seeds are proposal values requiring collision checks before any run.

## Static recording audit and next interface

In the inspected released implementation, the stepping-only observer runs before the common stepping cut, so a step subsequently killed by that common cut is observed. This does not restore tracks rejected during stacking or extend transport after a kill. Common-cut effects on later return trajectories remain part of the factorization audit.

The observer silently stops appending after its configured capacity apart from a warning. Generic target sensitive detectors have a separate geometry-configured `g4.stepsSizeLimit`; their overflow also drops rows with warnings. The event-status construction explicitly accounts for SimParticle overflow, step-limit kills and field-propagator kills, not these hit-collection truncations. A zero `StatusG4` is therefore insufficient. Preserve complete logs and verify both independent limits; any saturation is unresolved completeness, not a zero exit or permission to discard an event.

The released proton gun with `mean: -1` constructs one proton per generated event. `GenEventCounter` counts its event callbacks and writes an end-SubRun count; it does not count successful target crossings or independently prove completed transport. Reconcile requested IDs, persisted generator protons, original primary links, SubRun counts, completed events and failure logs before claiming POT normalization. Never turn an unwritten or truncated event into a zero-exit observation.

For initial implementation, prefer a root-coherent first-outward-crossing event library at the active target-material union, retaining every later crossing in the diagnostic ledger. This is a proposed interface, not a demonstrated autonomous transport boundary. Keep the target geometry available to the transport arm for re-entry and track all same-event companions; do not pre-replay stock-field returns if the exterior transport changes. Enumerate the material-member placements explicitly and exclude internal interfaces from union exits. A broader closed target-plus-support envelope is a distinct alternative requiring its own inventory, native membership and transport contract.

Before requesting simulation, resolve whether a default-off native recorder is required for paired endpoint times, unique pre/post touchable paths and copy histories, native boundary/track statuses, step ordinals, particle links and event-level attempted/stored-row counters. Geometry inventory and parent/material membership must be source/runtime coherent. The prepared all-step observer does not supply these fields and is not ready for library generation. An unapplied diagnostic source proposal now exists, but independent review and completion to build-ready source, a separately authorized user-run build, and subsequent native qualification remain required. This note authorizes none of those execution steps.

The [source/POT adapter plan](TARGET_SOURCE_ADAPTER.md) identifies actual generator, primary-helper, SimParticle insertion and written-product reconciliation boundaries. An unapplied/uncompiled [v2 diagnostic schema](TARGET_SOURCE_SCHEMA.md) now covers proposed input/link/insertion and in-memory collection checks; a separate [uncompiled original-product reader and structural join](TARGET_READBACK.md) now propose written-product/event/SubRun checks, but actual original-art verification and qualified POT remain undelivered. Native event abort is distinct from the limit-derived status and SubRun event counter. Recorder completion is not a persisted-output receipt.

The [native recording proposal](TARGET_RECORDING.md) specifies the proposed event/crossing/lifecycle ledgers, geometry inventory, exterior-feedback scope and staged approval gates. Unapplied source assets and structural checks are available in [native/](native/README.md); the draft is uncompiled, not build-ready and always library-ineligible. No launch-frontier or source-product/exposure admission is supplied.

## Library and validation gates

1. Establish inclusive recording and complete exposure before histogramming. Retain a root-coherent event library and zero-exit events alongside any summaries.
2. Split by independent source shards before choosing representation or binning; keep correlated descendants in the same split.
3. Publish separate charges/species and explicit overflow/underflow/unsupported bins. Condition response models only on measured/input variables; truth-only selections are diagnostics.
4. Validate transport independently first, then production plus transport at the named boundaries with paired per-shard uncertainty. A target library alone is not PS/TS closure.
5. Measure generation, material/decay transport and end-to-end throughput separately only after physics validation. Test-suite speed and selected-crossing counts are not POT throughput.
