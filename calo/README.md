# Calorimeter primitives (prototype)

This detector-specific package is additive and does not change the beam calorimeter-acceptance proxy or any chain default.
It is **not** an end-to-end calorimeter simulation or a validated replacement for stock Offline.

**R6 is paused at the 2026-11-05 checkpoint.** Start with [PAUSE_RESUME.md](PAUSE_RESUME.md) for current evidence,
branch locations, frozen-record caveats and the next gated task. Older review artifacts describe their preparation
time, not a request to rebuild or rerun completed controls.

## Implemented boundaries

- `calofastsim.response`: NumPy energy correction, expected PE per SiPM, nonnegative `std::lround`-style counts,
  bunch-time folding and inverse-variance reduction of **supplied** reco-digi group membership.
- `calofastsim.response_jax`: explicitly imported optional JAX twins; callers must enable x64 for source double
  arithmetic and preflight numeric validity on the host. No module changes global JAX settings.
- `calofastsim.propagation`: host float32 sequential CDF construction from regular depth/time histogram bins,
  validation and lookup with **supplied uniforms**; optional `propagation_jax` lookup.
- `calofastsim.hits`: vectorized NumPy stock-algorithm crystal/time grouping and weighted reductions of supplied
  fitted reco digis. Groups use first-member time, strict window splitting and total-crystal multiplicity;
  same-SiPM members and singleton time groups are allowed. Stable equal-time ordering is flagged, not claimed
  equivalent to unspecified C++ `std::sort` order. `scripts/check_hit_grouping.py` compares stock layouts/values
  without using lossy fit-equality associations or repairing them. No TemplateFit/digitization is performed.

The algorithms follow `Offline/CaloMC/src/CaloShowerROMaker_module.cc`, `CaloPhotonPropagation.cc`,
`Offline/CaloReco/src/CaloHitMaker_module.cc` and the corresponding data-product storage types.
All geometry, conditions, timing and histogram contents are supplied by the caller. No fitted response scales or
calibration constants are embedded. The `1e-6` CDF floor is part of the stock algorithm, not a fitted parameter.

## Exactness contracts

- Inputs are **compressed CaloShowerSteps**, with crystal-local position, not raw Geant4 deposits or global z.
  Birks selection chooses the stored visible energy; it does not compute Birks quenching.
- Means and energy corrections use float32 storage. Count rounding uses half-up for the nonnegative domain;
  ties-to-even and naïve float32 `floor(mean+0.5)` are not equivalent at all boundaries.
- Time folding uses `fmod`, a single conditional period shift and a lower-time cut only. Callers must not silently
  add an upper-time rejection.
- JAX compilation can contract float operations. Corrected-energy tests allow small arithmetic differences;
  this does **not** guarantee identical discrete PE counts at half-integer boundaries.
- Propagation samples a depth slice with truncation/clamping and returns a time-bin center using `lower_bound`.
  The caller supplies depth `crystal_length-z_local`; no depth interpolation or analytic travel-time stand-in.
- Reco reduction averages energy/time separately with inverse-variance weights, double accumulation and float32
  output. It does **not** find time groups, enforce crystal multiplicity, fit pulses, or build clusters. Parallel
  device scatter order is not guaranteed to match stock sequential summation bitwise.
- Matching seeds in JAX and CLHEP does not match realizations. Same-step exactness requires supplied realizations
  or a deterministic configuration in both arms. Photon statistics, noise, waveform extraction, TemplateFit,
  cluster finding and track matching remain stock-only.

## Tests and controls

```bash
PYTHONPATH=calo python3 -m pytest -q -p no:cacheprovider calo/tests
```

The unit suite uses synthetic fixtures and is **not stock execution validation**; tracked control scripts are separate.
Without JAX,
its cases skip cleanly. Package installation requires only NumPy; use the `jax`, `analysis` and `test` extras as needed.

`fcl/products.fcl` lists persisted products using the existing `FileDumperOutput` plugin.
`fcl/reco_control.fcl` runs existing stock reco-digi/hit modules on stock digis and exports the linked products through
EventNtuple. The latter is an explicit **Run 1A / Sim_best v1_1 control**, not a reproduction of every production campaign.
`CALO_EVENTS` bounds its event count; geometry/conditions and expanded FHiCL must accompany comparisons.
`art/CaloFastSim` contains a separately built, existing-art numeric exporter with scoped runtime checks; it is
**not available in the original pilot runtime** and does not establish a certified shower library.

`scripts/stock_control.sh` uses only an already-built area with the umbrella `container/offline/offline_uva.sh`.
Supply all `CALO_*`, `CVMFSEXEC_DIST` and optional `OFFLINE_IMAGE` settings explicitly and launch it in a CPU Slurm
allocation. It never builds, installs, or generates Geant4. `scripts/check_hit_weights.py` checks highest-cycle ntuple
entry count, linked multiplicities and float32 inverse-variance reductions, **not hit-finding efficiency**.
`scripts/audit_hit_links.py` diagnoses forward/reverse links, repeated member indices and identical fit signatures.
Duplicating a member can reproduce weighted values if two sensors have identical fits, so agreement of reductions
alone does not establish correct identity. The audit never repairs associations by time, distance or fit equality.

`scripts/census.py --out OUTPUT.json DID ...` uses the read-only metacat API after the operations environment is set.
The JSON contains exact catalog counts and an example's metadata; **catalog names do not verify detector contents**.
No private inputs, conditions or raw results are distributed here.

## Propagation follow-up

`PROPAGATION_CONTROL.md` separates actual-ROOT source/table conformance from public stock execution.
`scripts/check_propagation.py` audits TH2F regular bins, source-order CDFs and supplied-uniform NumPy/eager/JIT
JAX boundary probes. Passing this audit does not mean the private stock cache was observed.
The isolated `CaloPropagationDumper` calls unmodified public stock build/lookup with prescribed CLHEP draws;
its user-built one-event control has passed finite-probe comparison against NumPy and eager/JIT CPU JAX. The
no-build driver pins the table/source/plugin and linked stock library; the checker pins output hash and observed
file/histogram metadata. This does not directly observe private cache fields or validate production RNG.
No stock producer or earlier response-analyzer source is changed.

`TRAVEL_CONTROL.md` and `scripts/check_travel_control.py` define a separate same-saved-step travel-only arm.
Its stock execution has passed exact baseline event/step/condition/summary and pointer/SiPM NPE invariants, plus
all conditional float32-arrival support checks. It reuses the saved baseline and keeps empty/zero-PE responses.
The separately frozen post-support ideal-uniform CDF diagnostic is explicitly model-conditional, not production
CLHEP/RNG or stochastic-response certification.

`POISSON_CONTROL.md` and `scripts/check_poisson_control.py` define the separately executed PE-statistics-only arm
(travel off). Exact inputs/conditions/ordered summaries and no-travel time bits passed, with counts zero-filled for
**every accepted CSS times both SiPMs**, not only persisted positive readouts. Its prospectively frozen independent
ideal-Poisson CDF diagnostic was not rejected; this is not the actual RandPoissonQ/MixMax law. In particular stock
RandPoissonQ uses a skew-corrected Gaussian approximation for means at least 100. Zero-mean behavior has synthetic
coverage but no zero means occurred in this saved sample.

`COMBINED_CONTROL.md` and `scripts/check_combined_control.py` define the separately executed arm with both switches
on. Same-input/condition/summary invariants, the complete zero-filled count ledger and all conditional arrival-support
checks passed. Three prospectively frozen ideal-model families (count CDF, conditional travel CDF, and genuinely
joint count/mark contrasts) were not rejected; their conservative suite union bound is .03 under the ideal model,
not actual-engine calibration. No separate-arm sampled-count/seed equivalence is demanded. Production RNG, stock
sampler law, stochastic response, digitization/reconstruction and unbiased-library certification remain open.

## Missing before an end-to-end claim

1. Lossless shower-step and disk-entry tables, identity/ancestry and geometry/conditions provenance.
2. A root-coherent shower library with leakage/front/back material coverage, development/held-out split and an
   explicit out-of-support policy. A step-only library cannot infer the missing entry state or unseen showers.
3. Stock-built pulse/propagation caches and realized RNG/noise, then same-step response/digi controls.
4. TemplateFit agreement, complete hit grouping, cluster membership/position/time and track matching.
5. Paired CE shard comparisons (E peak/width/tail, time, position, E/p, match efficiency), cosmic MIPs, and separately
   measured stock/fast rates at the same boundary. Synthetic kernel timing is not generated-event throughput.

## Unfiltered library preparation

`calofastsim.library.pack` packs complete single-incident-root source events with correlated step offsets,
including empty roots. It requires audited unfiltered-retention, complete incident-census, zero-response and
completed-transport gates, a complete independent incident manifest, canonical source-artifact SHA256 identity,
and a one-to-one source-artifact/shard split. These declarations must be established by auditing the producer,
not merely set to true. Every incident has status-zero transport; nonzero-status records belong in a separate
failure ledger, never an empty response. Manifest omissions/extras and ambiguous multiple roots/crossings fail closed.
One designated first incoming crossing outside passive disk material conditions the **whole event**; later re-entries
and deposits in both disks stay correlated, not independent library roots. Crystal identifiers are global and must
be checked against the audited geometry by the exporter; the packer does not infer a disk from a crystal number.
Schema 2 preserves scalar-array types, zero-length offsets and JSON provenance in non-pickled NPZ-compatible arrays.
The packer is not yet a populated shower library or an interpolator.

`fcl/unfiltered_ce_library.fcl` retains stock CE events and raw truth, with both generator/detector filters disabled.
The bounded pilot configuration has been expanded against a pinned campaign; transport completion and completeness
remain separate gates. `scripts/expand_ce_pilot.sh` supplies the worker-visible upstream stop pool and expands it.
`scripts/unfiltered_ce_pilot.sh` requires explicit audited config/input/runtime hashes and a fresh output directory;
it runs exactly 2,000 transport events inside a separately approved one-CPU Slurm allocation using an existing runtime.
It saves raw art, resolved geometry, timing and product inspection, without building software. The output selector
retains G4-failure records even when the consistency filter stops downstream step making; these are not valid zeros.
`fcl/copy_stop_pool.fcl` and `scripts/prepare_stop_pool.sh` copy a bounded upstream stop subset without detector cuts;
they do not establish full source-pool coverage or POT normalization. Preserve the parent metadata and subset policy.
Removing the detector filter alone does not repair an incident census pruned by downstream detector activity.
`scripts/entry_census.py` is diagnostic only: a saved StrawGasStep's earliest time can hide later straw returns,
so a pre-entry-time cut is not a proof of shower-independent tracker selection.

## Export adapter and gates

`art/CaloFastSim` is isolated from Offline and existing pilot snapshots. Its user-built plugin has exported an
existing-art pilot without rerunning transport. Ordered numeric shower/raw-step/particle states and event-product
presence/counts passed scoped runtime audits, not a literal field-roundtrip or populated-library certificate.
See `art/README.md`. `scripts/export_pilot.sh` requires explicit input hashes and separate simulation/exporter
provenance, uses the simulation geometry search inputs and records completion; it never builds or transports.
`scripts/audit_export.py` reconciles exported event, registry and row counts plus productID/key links, including
GenParticle links separately from StageParticles. The configured raw-step product count is a required CLI input
(`--expected-step-products`), so a missing empty product cannot pass by shrinking its own registry. Array collection
indices must be contiguous, and status-zero parent graphs must be acyclic. `calofastsim.ancestry.assign` labels
references by exact event/ProductID/key ancestry to supplied single incident roots; it rejects unresolved identities,
cycles and references outside those subtrees, without time/proximity fallback. One source artifact is processed at
a time; descendant identity alone does not prove that a compressed step contains only post-entrance deposition.
The independent incident census and boundary/causal assignment remain mandatory; successful numeric reconciliation
deliberately does not certify zero-response completeness.
`scripts/validate_pilot.sh` is an existing-runtime, no-transport status census that requires a finished pilot receipt.
Its integer status log plus `scripts/audit_status.py` checks the finished art entry count and status population.
Passing that independent log as `audit_export.py --status-log` also requires exact exported event IDs/order/status.
`scripts/audit_geometry.py` checks numeric crystal/disk/VD identity, dimensions and frame consistency, explicitly
without certifying historical geometry equality or material coverage. `scripts/census_library.py` identifies
CE transport roots using the source-audited StageParticle parent reseating into the raw transport ProductID;
both original and reseated parent identities must exist. State differences are diagnostics, not identity matches.
Its shower ancestry, boundary-row and empty-collection counts are not independent incident/zero certificates.
Shower-positive events without direct generated-electron boundary rows are retained in a columnar gap report,
with exact boundary-particle ancestry and shower-ancestor identities; secondary crossings are not dropped or
silently substituted for a certified whole-event entrance.

`scripts/audit_boundary.py` classifies recorded shell momentum against source-defined outward normals and checks
necessary ancestry/time coverage for raw crystal steps and compressed showers. It also bounds float-stored birth
times, reports distinct entrance lineages per event, and compares raw/compressed counts and event/crystal energy.
It does not identify actual boundary-entry status or recover compressed constituent membership; late ancestral
crossings cannot supply pre-birth coverage. Multiple entrance lineages remain one correlated source event.
`scripts/audit_boundary_geometry.py` evaluates released disk/FEB/cable placement formulas with saved resolved
inputs. Its inter-disk cable-entry witness demonstrates why disk/FEB VDs alone are not a closed all-material
boundary. It also compares constructed shell z positions with exported GeometryService VD centers; metadata must
not silently override actual G4 placement. These helpers are diagnostics, not incident-manifest producers.
The current pilot supports response/compression controls but does **not** certify a single-entrance, whole-event,
all-material shower library. Empty response means empty at the stock crystal-SD/step-maker thresholds, never a
claim of literally zero deposited energy.

## Deterministic stock response control

[`RESPONSE_CONTROL.md`](RESPONSE_CONTROL.md) specifies a same-saved-step experiment using unmodified stock
`CaloShowerROMaker`, with Birks/LRU on and Poisson/travel time off. The isolated `CaloResponseDumper`
exports fetched conditions and exact saved-step links to observed PE times/counts and grouped energy/time.
It is **user-built and runtime-checked on an existing-art pilot**, with exact NumPy and eager/JIT JAX CPU-primitive
observed-output comparisons. This scoped result is not a full-chain/all-condition/device certificate. Any C++
change requires a completed user rebuild and new verification before loading it.
The transport schedule did not establish persisted event-window products, so an explicit fixed-phase stock
marker producer supplies controlled timing. This is not historical digitization replay.
`scripts/check_response_control.py` reconciles every event/saved step to the independent export and compares
persistent-key readouts, all PE times and ordered-member float summaries; NumPy or optional JAX backend.
It does not falsely label recomputed per-step means/energies as directly observed stock intermediates.
No stochastic, full-chain or unbiased-library certificate follows from this control or its synthetic tests.

## Native recording diagnostics and remaining gates

`recording/INTERFACE.md` defines the candidate native subtree/material inventory and separates initial entrance
sets from exterior feedback. A return from exterior scattering is an outcome, not a new independent incident.
The existing CE tracker-entry-plane endpoint is not a calorimeter hand-off. Coupled exterior/domain transport or
an explicitly broader validated environment is required before claiming a sufficient response kernel.

`recording/boundary/` and `recording/compression/` contain **default-off diagnostic review patches**, exact-source
hash manifests and temporary-copy applicability/structural checks. `recording/current_head/` identifies their
assembled isolated current-head source base, p110 environment and lifecycle refinement. Its partial-build helper
creates a three-package source-linked overlay against the exact matching published CI, leaving the full prepared
checkout intact; only Mu2eG4, CaloMC and CaloRecording are exposed to local build discovery. Native transition/birth/
terminal/event markers and actual raw-to-CSS emission membership completed an initial user build with source/
binary checks. Framework loading then passed, but zero-transport native initialization hit an inventory cap and
art rejected the analyzer-based manifest writers on trigger paths. The corrected always-pass filters and contextual
inventory diagnostics subsequently passed a second user build and fresh binary/source verification. Both writers
load as filters. Three-event single-root, strict-single and valid-empty generator-only jobs close successfully and
pass independent ROOT sequence/identity/double-state/metadata audits. Strict-empty, missing prior-process product
and generator-error checks reach their expected exceptions; their observed prefixes remain incomplete diagnostics.
The earlier v1 zero-transport inventory exhausted the cumulative 256 MiB file budget at placement 163,069, not
the separate StreamInfo budget. Compact v2 subsequently passed the user's incremental build, fresh source/binary/
linkage verification and zero-transport initialization. Its independent structural audit accepts 167,623 expanded
placements, 80 materials, 3,910 solids and 4,382 logical volumes: 175,997 rows and 49,035,617 bytes, within unchanged
caps. All 13 candidate roots occur once; 14,623 placements are union-selected. The job closes with one rejected
art event, zero G4 events and an empty boundary ledger. Observed VmHWM is 1,695.52 decimal MB versus 2,032 MB in
the earlier failed v1 attempt; this is not a general RSS bound or an equal-work benchmark. The PMR helper guard
remains min(file-cap, 256 MiB) of requested live allocation, not process RSS. Interned descriptions, expanded parent
references and the frozen-snapshot/exact-preflight design preserve all-world coverage. The successful closed-job
receipt is bound to inventory/config/source/binary hashes; the reader validates structure, not native geometry or
provenance authenticity. In particular, 1,143 expanded placements use unresolved solids, including the boolean
CalorimeterMother. Material composition, containment/navigation and raw membership invariance remain open.
`recording/audit_inventory.py` audits these existing outputs without launching art or transport.
Isolated Mu2eG4 relocations retain the same GSL CBLAS caveat as the exact stock backing; framework loading and run
initialization progressed, but no transport or normal-product invariance has been validated. Output caps and exclusive file creation fail closed. No shared source
checkout or saved-step-control plugin is changed. Current-head source preparation does not imply compilation,
physics equivalence or permission to launch. `recording/VALIDATION_PLAN.md` lists the separate build/native-
geometry/manifest/control and explicitly approved transport gates.

`calofastsim.recording` packs a separate diagnostic entrance-set schema, retaining correlated multiple incidents,
returns, born-inside cases, failed/incomplete events and empty offsets. It requires an independent event manifest,
explicit native frame/units/ProductID conventions and a bounded batch. It **always** marks library eligibility
false. No existing pilot VD rows are promoted into this schema and no library is populated. See
`recording/ENTRANCE_SET.md`. `recording/inventory/` supplies a capped native placement-tree exporter; unknown solid
topology stays metadata-only and replicas/parameterizations fail closed. `recording/manifest/` supplies separate
pre-generator EventID and pre-transport StageParticle manifests, explicit Mu2e-coordinate metadata and vectorized
external sequence/hash audits. The manifest writer has the bounded native evidence above; boundary transitions,
actual compression membership and native solid/containment validity remain unvalidated despite the completed
structural placement inventory. `calofastsim.boundary_ledger` strictly
parses framed JSONL with bounded reads and integer/lifecycle checks, reconciles the independently supplied manifest
and preserves incomplete/failed/unsupported reports. `audit_file` additionally verifies actual canonical manifest
and source-artifact bytes; that does not prove upstream independence or native provenance. Born-inside occupancy
is distinct from no entry, and p110 track-status semantics are explicit. Native solid/containment evidence,
manifest-to-boundary world transform/root reseating, raw-membership runtime validation and exterior-feedback
closure remain open. No adapter invents initial-sibling versus returning-descendant classifications.

## Raw-to-compressed replay and boundary proposal

`calofastsim.compression` reconstructs the inspected `CaloShowerStepMaker` algorithm from saved raw crystal
steps, the exact SimParticle parent forest and numeric geometry. It applies crystal-cell/start/end containment,
ancestor/crystal thresholding, depth buckets, first-time grouping, flush ordering and double weighted accumulation,
including the float constructor's time narrowing. `momentum_in` is the maximum constituent momentum magnitude,
not an entrance vector. Cell-union containment is a source-algorithm check, not native G4 material navigation.
`scripts/replay_compression.py` reads explicit expanded compression settings and compares ordered persistent
identities, counts and every exported shower numeric field. It can save a non-pickled diagnostic membership NPZ
with raw product/index identities, algorithmic bucket/generation and flush evidence. Existing evidence is not
overwritten. It returns failure when strict ordered counts disagree, even if a canonical row-multiset comparison
passes; that diagnostic sorting is never an identity join or association repair.

Equal-time source steps have unspecified C++ `std::sort` order. `--equal-time-order reverse` is a sensitivity
control, **not** an attempt to emulate stock sorting or a production default. Both alternatives stay flagged.
`scripts/compare_compression_membership.py` compares their partitions using raw persistent identities and exact
algorithmic ancestor/crystal/bucket/generation labels, not stored average time/position equality. Agreement of two
partitions and stored numeric summaries cannot replace independently emitted stock constituent associations.
Very late finite source times are retained and diagnosed, not clipped, discarded or called corrupt without evidence.

[`BOUNDARY_RECORDING.md`](BOUNDARY_RECORDING.md) proposes the closed-domain transition ledger, independent event
manifest, correlated entrance sets and optional stock compression membership needed before another pilot could be
requested. It remains a design/acceptance proposal with partial default-off diagnostics implemented and user-built;
native transitions, membership off/on invariance and library closure remain unvalidated. It is not permission to
build/package/run new transport.
