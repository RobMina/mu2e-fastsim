# Native-domain recording and calorimeter transport interface

Status: review draft, not an implemented transport interface or populated shower library.
The recording patches are optional diagnostics against a source-hashed snapshot, not a runnable campaign.
No boundary completeness, source/runtime equivalence, or physics promotion is implied.

## Decision: preserve the joint event; do not equate crystal with independent shower

The library draw unit is the designated generated-root event with its correlated exterior incident set and both-disk
response. Crystal number remains an output coordinate. Energy, angle and position are measured at an explicitly
owned native boundary for every exterior incident; species, momentum and time remain part of the state. Local
crystal coordinates are derived from the resolved native transform, never inferred from a VD center. Raw deposits,
future returns and successful-hit selection must not become conditioning variables.

Use a versioned entrance-set recording schema separate from single-entry library schema 2. Do not populate schema 2
by selecting the electron, highest-energy crossing, or first shower-positive crossing. Zero-length entrance and
response offsets are valid recording outcomes, not automatically calibrated zero-response draws. Keep failed,
unsupported, no-entry and born-inside cases, with explicit status. A production sampler/interpolator remains absent.

## Candidate domain and material ownership

For geometry auditing, the candidate domain is the union of native placed subtrees for both disks and all
calorimeter-assigned service material, not the union of disk/FEB VDs. The initial root inventory is:

| Native subtree roots | Construction responsibility |
|---|---|
| `CalorimeterMother` | disk solids, front/back/side passive structures, FEBs, internal `CaloCableRunCal_*` and inter-disk/end `CaloCableRunGap_*` |
| `CalCableRun`, `calCableRunFall` | external DS cable and connecting polycone, including nested cores |
| `CalIFBCableRun1`, `CalIFBCableRun2` | hall-side IFB runs and nested cores |
| `CalIFBCableRunEnd1`, `CalIFBCableRunEnd2` | IFB end boxes and nested cores |
| `CalIFBCableRunBottom1`, `CalIFBCableRunBottom2` | IFB bottom boxes and nested cores |
| `CalIFBCablePanelIn1`, `CalIFBCablePanelIn2` | inward panels and nested cores |
| `CalIFBCablePanelPOut1`, `CalIFBCablePanelPOut2` | outward panels, whose children are named `CalIFBCablePanelOutCore1/2` |

This is a **source-derived candidate inventory, not a native geometry certificate**. Enabled/version-dependent
volumes must be resolved from the actual construction; absence is allowed only when the frozen configuration
explains it. Require unique placed identities, full copy/touchable paths, materials, solid types/parameters,
transforms and exact parent relationships. Do not replace this inventory with a broad name prefix, whole hall or
DS vacuum: those would silently change the modeled physics. A core already nested below a root is not a separate
independent incident surface. All placements must be matched once; duplicate names require placement-qualified
selectors or rejection, not first-match selection.

`Mu2eG4/src/constructDiskCalorimeter.cc` places `CalorimeterMother` as a union of a full-phi disk annulus and a FEB
sector. It contains gap cable material but is not a full enclosing cylinder. `Mu2eG4/src/constructDS.cc` places
several external calorimeter cable structures directly in the hall, others in DS shield vacuum. Tracker services
are interleaved in the same longitudinal region. Source formulas identify the required audit, not containment or
non-overlap. Any decision to leave an external calorimeter cable upstream must be an explicit versioned scope
change with a transport owner; the all-material library claim must then be narrowed accordingly.

## Exterior feedback is part of the interface, not training input

A particle that leaves the domain can scatter or branch outside and return. A list of first exterior entrances
alone does not establish a Markov-complete standalone response kernel. In particular, the complete observed list
of returns is response-dependent: conditioning on it would leak future shower outcomes into the model.

The conservative interface target is **coupled transport**:

1. Exterior transport advances a root-coherent set to native outside-to-inside crossings, supplying double state,
   time, particle/root identities and geometry/domain version.
2. Domain transport returns deposits plus an ordered escaped-particle set, retaining parent/root/visit identity.
3. Exterior transport handles those escaped particles and any later returns; returns resume the same source event.
   The two owners never transport the same segment or draw a fresh independent full shower for a return.
4. A closed event reconciles the external queue and every active visit. Unsupported/suspended/aborted cases remain
   explicit and cannot be emitted as empty successful showers.

The library implementation of this coupled kernel is not supplied by the recording patches. A whole-event
marginalized kernel is an alternative only for a frozen, explicitly included exterior environment with a justified
upstream conditioning surface and held-out validation of feedback. That is broader than a calorimeter-local
library. Recording will quantify feedback/topology support before choosing that shortcut. Do not assert feedback
is negligible from current VD repetitions, negative pz, or absence of a recorded transition.

## CE chain hand-off

The current `beamfastsim.cechain` endpoint is a tracker-entry plane; `cesob` follows helices through a crystal map
with limited inner-ring material. It does not transport all tracker, FEE/front/back or external cable material.
Changing endpoint z cannot create a native calorimeter interface or upstream secondary history.

A future adapter must accept measured entrance sets from exterior transport, not just a CE at that plane. The
first validation path should use recorded Geant4 entrances and stock downstream response/reconstruction. Only
then connect an exterior tracker/CE transport that has independently validated branching and material ownership.
All source-shard development/confirmation splits apply to whole source artifacts and roots, never crystal rows.

## Recorder and review gates

- Begin an event ledger before any response selection; independently enumerate the input EventAuxiliary/source
  identity and generated roots outside this recorder. Reconcile begin/end markers against that manifest.
- Observe pre/post touchable membership, both endpoint states and step status without relocating a navigator,
  changing tracks, adding material, consuming RNG, or incrementing process counters. Birth/terminal records are
  distinct from exterior crossings. Record world exit, born-on-surface ambiguity and suspension explicitly.
- Record actual raw-to-CSS membership at stock emission in its observed constituent order, including rejection
  reasons and empty collections. Replayed partitions or average-value matching are not substitutes.
- Fail closed on write errors, duplicate identities, ambiguous placement selectors, row/byte cap exhaustion,
  incomplete event frames or inconsistent visit lifecycle. A partial artifact is retained failure evidence.
- Before build hand-off: rebase the patch to the chosen isolated current-head Offline area; check exact sources,
  mirrored dependencies and placement inventory. Never patch the existing response/exporter satellite or shared
  source checkout merely to make a draft compile.
- Before transport: user-completed build verification, geometry-only native inventory/containment/navigation
  tests, disabled-vs-enabled same-seed stock invariance control, explicit budget/runtime/site approval and a
  hard output cap. A small test is not a linear estimate of production runtime.
- Only after manifest/geometry/lifecycle/raw-membership gates pass may the new recording populate a separately
  reviewed entrance-set library. Existing saved-step response controls remain valid at their scoped boundary.
