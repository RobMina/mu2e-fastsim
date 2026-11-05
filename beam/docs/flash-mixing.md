# Flash diagnostics and background mixing boundary

The flash tools are explicit diagnostics, not a validated flash objective or a replacement for detector simulation.
They do not change the beam, conversion-electron, or cosmic defaults.

The [inclusive reference pilot contract](flash-reference.md) describes the owned,
unfiltered DS continuation, recording/normalization limits and acceptance gates.
It is preparation only, not a completed photon-flux or tracker-response reference.

## Pause checkpoint — 2026-11-05

R5 is paused. Photon births and geometry helpers are not validated exit flux; S2,
the 66-point comparison and tracker/CRV occupancy validation remain unfinished.
No owned Geant4 pilot or production-slot use has occurred. The separate pidfd-free
controller has 114 mocked tests but no host launch; its outer watchdog and
invocation-bound receipt are unimplemented. The latest combined focused software
checkpoint was 2029 passed / 10 skipped (overlapping earlier suites, not additive
or a full physics validation). See [the handoff](r5-pause-20261105.md),
[output/lifecycle gates](flash-reference-output-bound.md) and
[owned-child design](flash-reference-owned-children.md). Nothing resumes automatically.

## Existing compressed flash steps

`fcl/dump_flash_steps.fcl` configures the existing `StrawGasStepTableDumper` with the compressed beam generator tag.
It runs no transport. Use an already-built tracker analysis area. A completed job and the input/output artifact
identities must accompany any reuse: nonempty ROOT files alone do not establish successful processing.

```bash
PYTHONPATH=beam python3 -m beamfastsim.flash_ancestry input.root --out ancestry.json \
  --z-bins 0 4000 6000 8000 12000 20000 --r-bins 0 1000 3000 5000 10000
```

The output selects the highest cycle of each flat tree and joins particle identities within `(run, subrun, event)`.
The stock event and generated-event counters run beside the dumper; a successful end-job summary must accompany
any certified generated-event denominator. Per-subrun joint generator-species/birth-bin ledgers retain the
source grouping without asserting that different designs used paired source particles.
It follows the nearest **retained** strict photon ancestor, reports missing and cyclic ancestry, and counts
represented events separately from generated events. Compression can prune the original primary: neither a retained
root nor a generator link proves the original genealogy. Birth-position bins are diagnostics, not material labels.
The radial convention in this diagnostic is explicitly the dumped Mu2e coordinate origin, not the DS axis.
Do not identify a target or absorber by a broad z bin; use the source geometry and its coordinate transform.

`flash_census` performs read-only catalog requests on explicit dataset identifiers. Counts are catalog estimates;
metadata do not certify locality, detector payload completeness, generator normalization, or historical geometry.
Its optional namespace discovery lists one explicit namespace without expensive counts, filters dataset-name patterns
locally, then reads bounded metadata examples from selected datasets. Discovery and lazy-query failures remain errors
rather than empty datasets. A configured ntuple filename alone is not evidence that a photon analyzer was scheduled.

`flash_regions` uses the same ancestry joins and the represented target/absorber geometry to classify photon birth
coordinates into exact annular foil slabs, IPA, IPA rings, OPA, other, ambiguous, missing or no-photon categories.
It hashes geometry includes and preserves per-subrun energy accounting. This is a truth-level diagnostic, not an
Offline solid/material tag; supports, tilts, slots and detailed corrugation are not represented. Historical
geometry identity must be established separately. `flash_report --regions` compares the central-value region
ledger with certified denominators but never supplies paired uncertainties without matching source identity.

For cut-generated beam-catalog states use `fcl/read_flash_steps.fcl`: stock `StepPointMCDumper` reads the
collection without treating physical-volume copy numbers as virtual-detector IDs. It writes integer particle keys,
float hit values and a double-precision KinKal particle state; `StepPointsPrinter` preserves the event association
in its log. Its raw copy number is not a VD ID. The packaged tracker dumper may require absent detector products.

`fcl/read_flash_input.fcl` and `flash_input --stock` are retained as legacy diagnostics only. `ReadVirtualDetector`
assumes real VD geometry even for its global-column output; a cut-generated `BeamFilter:Beam` collection does not
satisfy that interface. Successful execution does not certify compatibility. Independent generic-reader values and
identity/order checks are required before reuse. `flash_timing` audits stored ntvd times, numeric creation codes and
original floating-point spacing without clipping or folding. At large absolute times the float32 representation
cannot resolve microbunch phase. `gtime` is proper time, not a generator-time offset. Neither finite values nor
an inspected coordinate plane certify pulse-time or generated-event normalization.

## Explicit device source records (partial source diagnostic)

`beamfastsim.flash_source_jax.slab_source` collects the optional core `lane_step_photons` records in a bounded
per-parent device buffer while running the existing slab step. The primary uses the same random draws. The return
contains an overflow count and unfinished-parent mask; `validate_complete` rejects either condition. This is a
slab-frame bremsstrahlung **birth** source, not the photon flux exiting a stopping target, and it requires the
explicit-photon core interface. It neither adds target-region geometry nor propagates any secondary. Positron
annihilation remains part of charged-primary energy accounting but is not in these brems records.

## Charge-aware target-region birth diagnostic (opt-in)

`flash_target_jax.target_source` reuses the CE device navigator with separate e- and e+ tables/charges and a fixed
parent batch per species. Its output is **primary-generated photon births in the represented target region**, not
the flux leaving a target surface. Bremsstrahlung and in-flight/at-rest annihilation have explicit mechanism tags;
delta-ray counts/energy are retained but delta rays are not propagated. No photons are attenuated or transported.
The bounded per-parent buffer reports dropped births, unfinished parents and loopers; `validate_complete` rejects
these numerical losses but does not certify missing physics. Initial below-cutoff positrons emit at rest once.

Vacuum step angles are limited for this diagnostic, including low-momentum turns; this does not solve every grazing
surface or re-crossing. The inherited navigator uses flat foil slabs with no rim/hole escape during a crossing,
radial approximations to shells, frozen layer fields and simplified absorbing boundaries. These approximations
require new low-energy checks before physics use. The default CE path is unchanged. Absolute source times remain
host float64; device state and birth times are elapsed times only. Slab slowing time uses the inherited mean-beta
approximation, not a precise time integral. Even double precision can lose nanosecond phase at extreme times.

`flash_target_run` requires a `flash_step_input` certificate bound to the exact exported state hash. The latter
checks collection-generic rows, event-qualified printer identities and legacy global Float_t columns for exact
equality before export. Its optional ancestry ntuple lacks subrun identity and must be explicitly omitted when
multiple subruns are present; this never authorizes cross-subrun ancestry joins. Saved rows are not generated-event
or POT normalization. Source-state time, event identity and charge are never folded, cut or replaced by a spectrum.
The runner preserves incomplete results and exits nonzero if numerical birth collection is incomplete. The full core
emission recorder replays the shared slab step behind a compiler barrier. Primary arithmetic bytes agree in
exercised regression configurations, not universally under arbitrary enclosing JAX programs or compilation contexts;
secondary outputs use the same primary draws but can differ by floating-point rounding. Defaults do not replay.
This extra computation must be included in source timing rather than inferred from the state-only CE benchmark.
When no live lane is inside a slab, the optional source skips the emission recorder and advances only the shared
slab RNG iteration. Active slab steps retain the same recorder and counters; this does not relax transport caps.

### Restartable bounded source

`initialize_source`, `advance_source` and `source_result` separate full raw transport state from the read-only
partial-slab projection used by reports. Advancement retains fixed per-charge row slots, species keys, shared slab
RNG iteration, frames, pending loss, counters and photon buffers; it neither refills lanes nor repeats initialization
annihilation. Exercised restart regressions are byte-identical across the tested chunk splits with bound runtime,
shapes, dtype, backend and enclosing compilation context; this is not a universal same-runtime guarantee.
A projected birth NPZ is not resumable.

`export_state`/`import_state` use a complete flat numeric schema and an exactly matching fresh runtime template.
The runner's `--checkpoint` writes a new hash-bound JSON/NPZ pair; `--resume` requires a new output checkpoint.
`--chunk-steps` writes a linked receipt after each bounded chunk. Bindings include source/certificate, settings,
geometry/includes/maps, code/physics assets and runtime/device kind. Corrupt hashes, foreign binding, mismatched
schema/dtype/shape/IDs/keys and inconsistent counters are rejected. No pickle is used and no artifact is overwritten.
Initialization of a restore template is not transport; raw checkpoint state replaces it before advancement.

`--max-steps` is work allowed in this invocation, not a replacement for prior cumulative iterations. The chunk loop
ends when no parent is live or a birth buffer overflows. Path-loopers remain recorded and fail `validate_complete`;
`transport_complete` only means no live/capped parent or dropped record, never an unbiased exit-flux certificate.
A higher iteration allowance does not cure lateral geometry, omitted cascades or missing photon attenuation.

`flash_checkpoint_report` audits a complete saved initial-to-child checkpoint chain without transport. It checks
state hashes, manifest-supplied binding consistency, parent-hash links, fixed source/species identities, cumulative
iteration increments, nondecreasing counters, byte-preservation of earlier photon records and frozen terminal
primary states. It reports each chunk's fates and the final fates of initially live parents. This is internal-chain
integrity, not independent certification of the binding's code/geometry/source claims or a physics-validity gate.

`flash_transport_report` inspects a saved diagnostic without rerunning transport. It retains columnar source
identities for capped parents and path-loopers, final state/crossing/adaptive-step counters, nearest represented
foil faces for capped rows and nonfoil births. Its optional reference comparison checks every saved array byte-for-byte. A small angular vacuum
step can exhaust the iteration cap before reaching the path boundary; neither a finite state nor closed photon
energy accounting permits unfinished parents to be silently dropped or relabelled complete.

### Finite-foil geometry prerequisites (not integrated)

`flash_foil_boundary.first_transition` and its optional JAX twin find the first material transition for a straight
ray through an exact finite annulus. They support both entry and exit, faces/rim/hole, surface starts and finite
path limits; unresolved tangencies/corners/conditioning are distinct from a no-hit. Optional `after_surface` names
an already-consumed exact transverse start root and uses forward ownership; it excludes only that zero root.
The curved twin certifies departure using monotonicity and clearance bounds before searching later recrossings,
not a finite epsilon advance. Wrong tokens, approximate rather than exact receipts and wrong ownership refuse. This ray primitive must not be
used as an exact helix or condensed-MSC boundary solver. The source still uses the legacy foil slabs.

The core optional `lane_step_emissions(first_exit=..., finite_lanes=...)` accepts a traceable first-extra-exit
callback on each actual leg before loss/hard-event decisions. It flushes pending loss at an exit, preserves ordinary
range-out, and disables the z-only trapped shortcut for finite lanes. The caller still supplies geometry and refusal
status. This hook is not enabled in the charged source.

`flash_foil_query` supplies a standalone pure leg adapter using matching gathered foil IDs, retained frames,
field and charge, with explicit unresolved status. The core currently has **no transactional status/rollback API**
to consume this pair safely; the distance-only hook must not interpret unresolved geometry as a resolved no-hit.
A proposed extension was rejected after strict runtime-dependent no-hit byte regression failed, so adapter/core
single-step tests skip unless a future compatible interface is supplied. This adapter is not enabled in the source
and is not full navigation, a numeric-receipt protocol, or physics validation.

`flash_foil_helix` is a separate constant-vector-field first-transition experiment, including tilted fields. A
chronological dyadic search bounds interval clearances to avoid skipping same-sign endpoint excursions; isolated
monotone brackets yield approximate intersections. Tangencies, corners, sub-resolution intervals and exhausted
search budgets refuse explicitly. Its numerical guards are not outward-rounded interval arithmetic or a universal
floating-point proof. Zero-force lanes delegate to the ray primitive. Neither primitive is enabled in transport.
Global-coordinate float32 guards can exceed a thin foil's thickness and refuse most crossings; a finite coordinate
is not evidence of adequate spatial resolution. Any local-coordinate alternative needs its own guarded transform,
field and frame checks before adoption, not a reduced tolerance chosen to suppress refusals.

`flash_foil_frames` builds immutable unique-foil tables separately from material-thickness types. Sentinel-safe
gathers retain foil identity, and proper-rotation sided frames keep the actual global point and depth for side
entries. They do not move side entries to a foil face or rebuild the frame after scattering. Invalid identities,
nonunit directions, out-of-solid positions and unresolved depth refuse. Coordinate roundtrips are not bit-exact.
`flash_foil_navigation` broadcasts first-transition queries over lanes and foils, including rim/hole candidates.
It refuses unresolved competitors and near ties, retains the selected unique foil ID, and never infers material
identity from equal thickness. These remain standalone adapters, not the source navigator.

Finite foil transport additionally needs curved first-crossing geometry, unique foil identity, sidewall re-entry at
the correct depth, and unresolved-navigation accounting. Layer type is not foil identity: equal-thickness foils
must retain distinct radii and positions through gathered slab lanes. The slab callback must transform each actual
MSC leg into the global foil frame and preserve its matching local field and charge. Vacuum must compete all
relevant foil face/rim/hole entries with the existing navigator boundaries. A first-transition primitive's backward
surface ownership is not a post-transition re-entry protocol: the navigator must carry the forward side and exclude
only the already-consumed zero-distance root, without an epsilon displacement or skipping a later recrossing.
The [navigation evidence and transaction design](flash-navigation.md) separates exact continuation, numerical
geometric evidence and material-transport restart. Host-only `flash_foil_receipt` and `flash_foil_follow` can
retain/recompute initial crossing evidence and search one later event on the unchanged anchored trajectory,
without rebasing a particle at an approximate endpoint. Their outputs always refuse material-restart authorization;
missing/unsupported/unresolved evidence is never certified clear. No new transport-status interface is implemented.
The exact `after_surface` adapter alone does not meet the numeric-receipt gate: independently propagating a
bisection result can leave nonzero surface residual. A solver-owned path bracket, uncertainty and forward-side
receipt must bind the same trajectory and frame before such endpoints can be reused. Arbitrary near-surface
coordinates, geometric snapping or tolerance relabelling are not accepted receipts.
Unresolved search must survive as an explicit invalid/incomplete outcome, never as infinity interpreted as vacuum.
An exit-only endpoint clip or post hoc relocation of photons is not a substitute. Even correct intersections retain
the inherited random-hinge scattering and endpoint stopping conventions, requiring step-refinement/reference checks.

## Proposed mixing contract (not implemented or validated)

The reference-preserving boundary is a **complete detector-step frame before detector response**, not a collection
of independent digitized hits. Keep each source event's tracker, calorimeter and CRV payload together: sampling
detectors independently destroys inter-detector correlations. Every frame has the following contract:

- explicit source stream, event identity, artifact hash, source configuration and geometry/conditions identity;
- compressed tracker `StrawGasStep`, calorimeter `CaloShowerStep`, and `CrvStep` payload, with closed SimParticle and
  GenParticle references, including empty collections rather than silently dropping empty detector payloads;
- generated-event/POT normalization, selection/filter fractions and prescales from the source production chain;
- source pulse-time convention and timing products, with an explicit indication of offsets already applied;
- donor frame identities, reuse accounting, independent RNG stream keys and a fixed permutation/resampling policy.

For a single primary event first draw the proton-bunch intensity with the appropriate primary-dependent law.
Conditional on that common intensity, draw each background-stream multiplicity with its configured Poisson mean.
Stock efficiency chains are obtained from production conditions/bookkeeping, not from the number of saved events.
Use stop, electron, muon and neutral streams separately; retain the stock stopped-muon daughter exclusion in the
muon flash to prevent double counting with stop pileup. Merge steps before response and digitization, using the
same per-channel detector response as the signal. Preserve the source event's joint detector correlations.

Early flash and ordinary flash are not interchangeable. Their event selections, prescales and compensating mixing
weights must be part of the provenance. A gate on a qualifying step is not automatically a cut on all saved steps.
Do not apply an additional proton-pulse offset or modulo-time fold unless the input convention calls for it.

Digi overlay is a separate approximation: concatenation does not model same-channel deadtime, saturation or pulse
interference. Trigger-selected mixed output is not an unbiased background-frame pool.

## Required tests before physics use

1. Exact identity remapping and reference closure for zero, one and multiple donor frames; no cross-event joins.
2. Conditional Poisson means/variances and correlations through the shared proton intensity; beam-primary
   intensity weighting versus nonbeam-primary weighting.
3. Pulse and clock-phase conventions at gate and microbunch boundaries; no double time offset.
4. Detector-OR source selection, early/main prescale normalization, frame reuse and double-count exclusions.
5. Paired per-shard tracker and all-sector CRV occupancies against stock mixing on the same finished primary events;
   calorimeter occupancy and shared-channel response checks before any trigger comparison.
6. Persisted empty frames and timing/normalization provenance, with generated-event denominators independent of
   nonempty-step or triggered-event counts.

Choose implementation variants on a development subset and confirm the selected one once on held-out inputs.
The ancestry diagnostic and a bremsstrahlung source primitive do not implement Compton, photoelectric or pair
conversion, secondary-electron cascades, photon attenuation, or a full shower response.

## Response-library denominator gate

An event-selected detector-step sample is sufficient for the deposited-energy numerator with a certified generated
event counter, but not automatically for the inclusive number of emitted photons. Full retained ancestry in each
saved event does not restore photons in rejected events. Nor does retaining every generation imply retaining every
particle in a saved event: detector-step compression can keep only the contributing ancestry closure.

A per-photon unconditional response library must therefore pair response numerators with an independently complete
photon-birth or boundary-state exposure, including zero-response photons, and certify how both are selected and
normalized. Dividing deposited energy by the retained photons in a detector-selected tree estimates a selected-sample
conditional quantity, not an unbiased response per emitted photon. Generated-event counts alone cannot replace that
photon exposure. Audit existing inclusive simulation or analysis outputs before requesting new transport. Do not fit
a response scale to design-point objectives to compensate for an unavailable denominator.
