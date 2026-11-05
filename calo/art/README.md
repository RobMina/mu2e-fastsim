# Calorimeter numeric export adapter

`CaloFastSim` is an isolated muse satellite; it does not modify Offline or the existing pilot runtime.
`CaloLibraryDumper` writes numeric TTrees, preserving every input event (including empty responses),
raw sensitive-detector steps, generated StageParticles, full event SimParticle collections/parent identities,
and compressed CaloShowerSteps with their original order. A missing shower product after a G4 failure is flagged,
not turned into a valid zero. No filtering, matching, transport or response model runs in this adapter.

**Built and runtime-checked for one existing-art pilot, not a certified lossless library.** A user-built coherent
newer exporter has read the original simulation art. Event identities/order/status were independently reconciled
against original-art PrintModule output; configured product counts, ordered row indices, persistent links and
status-zero parent forests passed the numeric audit. Exported geometry frames and dimensions are self-consistent.
These checks do not certify every field's round trip, historical geometry equality, exterior material coverage,
incident completeness or zero responses. Synthetic tests alone cannot certify the C++ adapter.

Header contracts must be checked in both releases. No plugin is in the original pilot runtime; shipping one to
UVA requires separate runtime-tarball approval. Preserve the immutable pilot snapshot and simulation provenance.

## Additional response analyzer: scoped runtime control

`CaloResponseDumper` and `fcl/response_control.fcl` are a separate deterministic same-saved-step control,
**user-built and runtime-checked** after independent new-plugin linkage/timestamp checks. Observed deterministic
outputs compare exactly with NumPy and eager/JIT JAX CPU primitives on one existing-art pilot, not all conditions
or the full chain. They do not inherit the library dumper's gates or certify stock per-step intermediates.
Stock response and the library exporter source are unchanged; mirrored SCons/CMake dependencies include the
condition-service calls. A C++ edit requires a fresh full user rebuild and verification before execution.
See [`../RESPONSE_CONTROL.md`](../RESPONSE_CONTROL.md) for exact-pointer observations, explicit controlled
marker timing, comparison contract and limits. Direct per-step response intermediates are not stock emissions.
The statement above that no response model runs applies to `CaloLibraryDumper`, not the separate control job.

## Numeric contract

- `events`: exactly one row per input event, full run/subrun/event identity, raw G4 status and counters,
  source-collection product IDs, presence flags and row counts. Status-zero is distinct from stock's status<=9 acceptance.
- `particles`: every SimParticle in every event collection, productID+key, parent/generator persistent identities,
  stage, PDG/process, start/end positions, four-momenta/times and termination detail. This is a numeric physics export,
  not literal ROOT serialization of every private persistent member. No key-only cross-collection matching.
- `generated`: every StageParticle's state and persistent parent identity, without narrowing double state variables.
- `genparticles`: each event GenParticle collection, separately from StageParticles; SimParticle generator links
  join productID+index here rather than to the CE StageParticle rows. Missing links fail for status-zero transport.
- `steps`: ordered raw StepPointMCs from explicitly named products, with product identity, instance kind and row index,
  pre/post position and momentum, time/proper time, energy components, length and terminating process.
- `showers`: every compressed CaloShowerStep's row index, crystal, time, position, Birks/G4 energy, scalar momentumIn,
  constituent count and persistent ancestor identity. `momentumIn` is NOT an entrance three-vector.
- `crystals`, `disks`, `virtualdetectors`: actual geometry service output, global/disk frames and sizes/transforms.
  Geometry/FHiCL/release/input checksums accompany the file externally; decoded numbers alone do not certify geometry.

A missing StatusG4 is a hard failure (unknown transport), not a retained valid event; partially written tables are
not accepted unless exporter completion and exact input-event reconciliation pass. On nonzero-status events,
missing generated/transport/step/shower products have explicit flags or registry absence; no valid zero is inferred.

A table producer is not an unbiased library by itself. Reconcile all event counts against the finished art,
check every pointer identity and ancestry edge, and derive an independent incident manifest before assigning steps.
Keep nonzero statuses in a separate failure ledger. Choose first **incoming** crossings outside passive material,
including radial/FEB boundaries and both disks; unsupported geometry/crossing cases fail closed.
Whole-event causal assignment, leakage labels, split independence and zero-response certification remain Python-side gates.
`calofastsim.ancestry` checks acyclic parent identity and descendant membership but does not recover pre-compression
constituents or prove their time ordering relative to a chosen crossing. VD centers/frames alone omit shell radii,
FEB phi faces and actual material coverage; they are not a closed exterior-surface certificate.

## Build hand-off (only after explicit approval)

Prepare this isolated `art/` satellite with an explicitly pinned, coherent backing/envset pair. The exporter can
use a newer pair than the simulation; it must not force new external dependencies onto incompatible old libraries.

```bash
CALO_BACKING=/absolute/pinned/backing CALO_ENVSET=pNNN bash calo/art/setup_area.sh
```

This preparation checks the backing's Offline envset recommendation and matching AL9/prof/e29 build directory,
creates the backing link and explicit `.muse` envset, and prints the user command. It never runs environment setup
or a build, and refuses to replace a different existing choice. `CaloFastSim/` remains a real
directory because muse's source walk does not follow symlinked repositories. The user, on the designated build host,
runs a **full** `muse build -j 32` after environment setup. Do not run a package-target build.
Both `CMakeLists.txt` and `src/SConscript` enumerate the plugin's direct dependencies.
Verify backing/envset/plugin timestamps, then run a tiny existing-art export and inspect rows/identity/counts.
Do not launch another Geant4 job to test an exporter. A shipped runtime tarball needs separate approval.

## Additional propagation analyzer: scoped runtime control

`CaloPropagationDumper` is an isolated one-synthetic-event control of the **public unmodified stock**
`CaloPhotonPropagation::buildTable/propTimeSimu`, supplied through CLHEP `NonRandomEngine` with counted draws.
It exports observed ROOT bins/centers, explicitly reconstructed probe thresholds, and actual stock returned times.
It does not access private stock cache fields or modify stock producers. See `../PROPAGATION_CONTROL.md`.
The user-built plugin has passed a finite-probe comparison against NumPy and eager/JIT CPU JAX; it does not
validate every possible depth/uniform or other device. Earlier response/library exporter sources are unchanged.
Any future C++ edit still requires a **new user full build** before loading. `propagation_control.sh` requires fresh
verified binaries, pinned CaloMC linkage, source/table/plugin hashes, a fresh output directory, and records effective
FHiCL and geometry. The checker pins the output hash and observed resolved file/histogram metadata. No transport,
production RNG, straight-line-mode or stochastic-response closure is implied by this control.

The existing response analyzer is also reused unchanged by `travel_control.fcl`. That travel-only stock arm has
passed exact saved-baseline count/identity/condition/summary invariants and conditional float32-arrival support;
see `../TRAVEL_CONTROL.md`. It is not eventwise random-realization equality or production RNG certification.
The separate `poisson_control.fcl` also reuses this unchanged analyzer, enabling only PE statistics. Its same-CSS
input/condition/summary/no-travel-time invariants passed with a complete accepted CSS/SiPM zero-filled count ledger;
see `../POISSON_CONTROL.md` for the frozen ideal-Poisson diagnostic and stock Gaussian-approximation caveat.
The separate `combined_control.fcl` enables both switches with the same analyzer and saved inputs. Its exact
input/condition/summary and count-ledger/arrival-support hard gates passed, with three prospectively frozen
ideal-model families not rejected; see `../COMBINED_CONTROL.md`. Shared-engine draw consumption means separate
arm realizations need not match. None of these arms certifies the actual engine law or downstream digi/reco.

## Reading older simulation with a newer exporter

Record the simulation release/configuration/geometry separately from the exporter backing/envset. Reusing the
original art does not regenerate its transport with the new release. After building, first validate a small
existing-art export, then the complete file: exact event identities/counts, product registry and collection counts,
resolved ProductID/key links, parent graphs, ordered rows and source field precision. Header equality is useful
pre-build evidence, not a ROOT schema-evolution or runtime compatibility certificate.

The geometry trees describe the exporter's GeometryService, not automatically the simulation's geometry. Run the
exporter with the saved simulation geometry configuration/search inputs, compare resolved values against the
pilot's saved configuration, and validate actual crystal/disk/VD tables before accepting entrance or local/global
coordinates. Hashing only the top-level geometry file does not audit its include closure or geometry-maker changes.
If parity fails, preserve the source-state export as diagnostic and keep geometry-dependent library gates closed.
Current-stock response/reconstruction is a separate target: source changes since the pilot need explicit audit
and new same-input controls; historical control success must not be relabelled current-stock closure.
