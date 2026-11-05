# Inclusive flash reference pilot

This is a prepared, opt-in reference design, not a completed simulation, a photon
response library or an exit-flux validation. No production defaults change.

## Pause checkpoint — 2026-11-05

No owned Geant4 pilot or production-slot use has occurred. The user-built exporter
and confined synthetic persistence checks do not close the
[saved-output/lifecycle gate](flash-reference-output-bound.md). Trial 22482090
failed with ENOSYS before releasing its root gate; cleanup remains unknown.
Self-only 22484355 completed with negative pidfd capability, not lifecycle success.
Pidfds are ruled out. The separate [owned-child prototype](flash-reference-owned-children.md)
has 114 mocked tests but no host launch; its outer watchdog and invocation-bound
receipt are unimplemented. All guard and physics-completeness claims remain false.
See [the pause/resumption handoff](r5-pause-20261105.md) for evidence, remaining
physics work and the conditional resource approval. Nothing resumes automatically.

## Purpose and scope

Use an owned unfiltered detector-solenoid continuation to measure both the photon
source and its associated tracker straw-gas deposition. Detector-selected output
cannot supply an unconditional photon denominator. Preserve every successfully
processed source event, including events with no photons or tracker deposition.

The initial sample is an instrumentation pilot, not a precision tracker-response
measurement or a substitute for held-out design validation. A small sample can
have zero tracker deposit; that is a valid result, not a reason to select events.

## Source assignment

Read an explicitly hashed existing DS-entrance catalog with `RootInput` and use
`BeamFilter:Beam` directly as `StepPoints` primaries. The input volume metadata is
`compressPVBeam`; let the runtime determine the continuation stage from that
metadata. No resampler, random initial skip, wrapping, pulse offset, time fold,
prescale or generator counter is scheduled. Preserve the original event identities
and every input StepPoint in order. This is a continuation of an existing upstream
simulation, not a new proton-to-detector simulation.

Replay the same ordered source-event prefix in each design. Verify the exact
EventIDs, input ProductID/particle keys, state ordering, charge, position, momentum,
global/proper time and new-stage parent links before calling the source assignment
paired. Independent downstream seeds do not make the showers identical. Repeated
input hits for one particle are separate transported primaries; do not silently
deduplicate them. Keep non-electron species if present and report them separately.

The normalization is initially **per selected source event and per DS input
state**, with charge-specific counts. The source catalog's inherited upstream
`GenEventCount` is not the number of events transported by this prefix. Neither
that count nor a newly added draw counter certifies POT normalization. The template
therefore schedules no `GenEventCounter`, which also conflicts with inherited
SubRun counters on a direct read. Full-pool/origin weights require a separate
provenance calculation; a contiguous prefix is not certified representative of
the entire pool. Historical board values have a different source campaign and
cannot be expected to close under an unverified normalization conversion.

## Recording contract

`fcl/flash_reference_raw.fcl` uses stock `Mu2eG4`, with empty stacking/common
cuts and a photon-only non-killing step recorder:

```fcl
Mu2eG4SteppingOnlyCut: {
  type: intersection
  pars: [
    { type: pdgId pars: [22] },
    { type: constant value: false write: "r5PhotonSteps" }
  ]
}
```

The ordered intersection stops before the recorder for non-photons. For photons,
the constant child records the step and then returns false. Thus the intersection
never kills a track. `write` belongs on the constant child, not the intersection.
This is supported by the stock cuts implementation and its all-step study example;
it still needs verification against the actual installed runtime before transport.

Retain all produced/input products in an unconditional `RootOutput`, without
`SelectEvents` or response/status filters. Preserve:

- the full new `g4run` SimParticle collection, including copied input ancestors,
  tracked photon births, terminal states, creation/stopping processes and Ptr links;
- `g4run:r5PhotonSteps`, including zero-edep photon steps, with pre/post position
  and momentum, pre-step global/proper time, path length and process;
- raw `g4run:tracker` steps from **all depositing species**, calorimeter/CRV/VD
  products, source generators, SubRun physical-volume metadata and `StatusG4`;
- the event ledger, including zero-photon and zero-tracker events.

The initial raw pilot schedules no StrawGasStep/shower/CRV aggregation or truth
compression. Aggregation can be applied later to these saved raw products without
rerunning Geant4. Select the new-stage product explicitly: keeping original and
extended SimParticle branches must not double-count copied ancestry. Birth rows
must be limited to newly transported tracks, not all copied upstream photons.

"Inclusive" means tracked photons under the declared physics/production cuts,
not arbitrarily soft secondaries suppressed by Geant4 production thresholds.
The initial range production cut is explicitly 0.010 mm. A production-cut
convergence check is separate and must precede claims of cut-independent response.
The physics list, regional cuts, radioactive-decay settings and field integration
settings must be recorded from the installed expanded configuration.

## Geometry and boundary

`beamfastsim.flash_reference_prepare` creates a fresh configuration bundle. It
expands geometry includes against one explicit runtime root, preserves repeated
include order, rejects cycles/root escape, and hashes every source and prepared
file. It explicitly sets geometry `g4.stepsSizeLimit = 0` to disable sensitive-
detector hit truncation. Frozen expanded geometry is used by the pilot; a moving
`geom_common.txt` alias is not its provenance. Separately bind/hash the field
configuration, map payloads, executable, libraries, environment and worker image.
Hashing preparation inputs alone does not certify those runtime identities.
The generated job embeds the exact copied template text rather than resolving a
bare template include from an uncontrolled FHiCL search path. Standard release
includes still require the pinned runtime and its expanded configuration.

The two transverse stopping-target VDs are **not a closed surface**. Define and
freeze an enclosing cylinder or the constructed target-mother cone with endcaps,
using actual effective dimensions. Audit photon step segments against that
surface, retaining incoming as well as outgoing crossings, surface ID and crossing
ordinal. Report both crossing flux and unique exiting photons. Photons born
outside may enter and exit; target-born photons may never escape. Birth and exit
ensembles must remain separate. An S2 response assignment must also define a
non-overlapping ancestry frontier: summing a photon response and the response of
its separately counted escaping descendant double-counts a shower. Preserve all
raw crossings for that audit; do not infer additive independent response merely
from unique photon keys. Returning photons and charged particles crossing the
boundary are explicit factorization tests/residuals, not silently omitted.

The stock step record lacks an explicit region-crossing flag and post-step time.
For photons, a straight segment may support crossing reconstruction only after
step/process, endpoint, corner/tangency and timing validation. Do not claim exact
online boundary tuples or interpolate charged paths as straight lines. Retain
ambiguous segments explicitly. If exact online volume-crossing/post-time records
are required, stop for an isolated recorder change and user build.

No flat generic photon ntuple is yet certified: the stock generic dumper lacks
complete event-qualified post-step information, and the ordinary ancestry ntuple
can omit SubRun identity. Raw art remains authoritative. A typed event-qualified
export through the analysis framework is implemented in the isolated art package.
Each declared runtime requires build verification, analytic fixture extraction,
persisted reread comparisons and existing-source checks. Successful FHiCL
expansion does not supply that evidence. The inspected
`StepPointMCDumper`, `SimParticleAnalyzer`, EventNtuple MC summaries and stock
printers do not collectively supply that complete export. Printers either omit
fields or impose fixed decimal precision; an event-header join cannot restore
missing fields or discarded precision. See the [export contract](flash-reference-export.md).

Preserve each product's stored precision, rather than promising precision absent
from the persistent type. `StepPointMC` stores pre-step global/proper time as
double, but deposits and path length as float; it has no post-step time.
`SimParticle` stores start/end positions as double, momenta and global/proper times
as float. Widening those floats to double does not recover the Geant4 values.
Source-hit identity comparisons must retain the original double StepPoint times;
new-primary state comparisons must explicitly account for the destination's
storage conversion. Do not weaken identity checks or fold extreme source times
to conceal a storage difference. The photon recorder's volume number is a touchable
copy number, not a unique index into the SubRun physical-volume table.

## Installed-source evidence, not transport validation

Read-only inspection of the closed configuration-only preflight 22466576 checks
saved v13_41_01 sources rather than a paused workspace mirror. Its geometry
hashes remain low c75c0fc5823a8ad2ffeb857786f243f3f7de022c4e9a1c3feb74f6443de026e0
and high 8c6e1d649f6c9d9786bcfb169005a4bdf8857eb63e11f818d04092e8677093e3.
These freeze configuration text, not constructed solids or an enclosing surface.

Saved Mu2eG4Cuts.cc SHA256
46a268ced3c8861fbf828f9ccc0bb3b141da2ff7a8c4a79fe11368e93b67dbff
shows IOHelper::addHit recording both positions minus mu2eOrigin, both step-point
momenta, pre-step global/proper times, the pre-touchable copy number, step length
and post-step ProcessDefinedStep code. Overflow prints a warning without adding
a row. Intersection short-circuits on its first false child; the constant child
adds its hit before returning false. Saved Mu2eG4SteppingAction.cc SHA256
bcaac31fbf84bd6bf6bcf9c8c1b9614415a71e6cf2e4b4e06a008cbd03c7b045
calls the stepping cut before the common cut. These support the intended
non-killing photon recorder and coordinate interpretation at source level.
They do not demonstrate an actual photon step, runtime binary/source identity,
straightness, complete recording or constructor success.

The saved tracking source separately stores a SimParticle end position from the
track but obtains end momentum and times from pre-post-step-doit helper state.
Do not equate that terminal state with a StepPoint post-state or reconstruct a
missing step post-time from it. No online boundary status/touchable path is added
by the exporter. The preflight marker explicitly records transport_events=0.

## Spatial exit diagnostic and causal response boundary

`flash_photon_surface.box_segment_crossings` supplies only the first geometric
piece: vectorized straight-segment intersections with all six faces of an
explicitly supplied axis-aligned box. It retains both roots for an outside-to-
outside traversal, direction and face bits. It refuses the whole row for
endpoint, edge/corner, coplanar, near-parallel or numerical ambiguity, retaining
NaN slots and an explicit status rather than converting refusal to no hit.
Engineering floating-point guards are not exact predicates or a completeness
certificate. There are no detector-coordinate defaults, event selection, native
times, history deduplication, response weights or transport calls.

`flash_photon_chords.diagnose_photon_chords` is a separate geometry-independent
prerequisite diagnostic. It retains one aligned row per input with chord length,
signed chord-minus-stored-length residual, positive finite float32 length ULP,
pre-momentum direction cosine and transverse sine. Input/numeric reason bits and
NaNs describe individual unavailable metrics; there is no valid/complete flag,
straightness cut or automatic crossing admission. Vector conversion is also
conservative: integer components outside the inclusive exact float64 range
[-2**53,+2**53] and wider floats changed by float64 round-trip are unavailable
with a numeric reason, never fabricated zero chords. No check can recover input
precision already lost before this call. Bad momentum does not discard
a measurable chord, and a backward chord retains negative cosine. It requires
original float32 stored lengths, not widened values claimed to have that storage
spacing. The caller must first bind the explicit photon product/PDG and event/
new-stage particle identity. No histories, time interpolation or endpoint repair
are performed, and a zero diagnostic bitmask is not a physics certificate.

Before applying it, freeze an envelope from the actual constructed geometry of
each owned design, in the same Mu2e-global coordinates as the recorder. Declare
whether it encloses only foil material or the full target assembly, including
supports; audit other enclosed material. A nominal target-mother radius is not
sufficient: Offline can construct a cone and derives mother extents from other
components. No numerical envelope is yet certified for the owned reference.
Validate straight photon chord/path/pre-momentum consistency and installed
recorder semantics before interpreting segment intersections as spatial exits.
The missing post-step time precludes claiming native timed crossing tuples.

Next build event/product/particle-qualified histories with crossing and incoming/
outgoing ordinals, explicit entry versus reentry, shared-endpoint ownership,
boundary births/deaths, continuity and refusal ledgers. Report crossing flux,
unique photons and first exits separately, split born-inside/outside and retain
inside births with no escape. Normalize only on complete source-event and DS-
state ledgers, not inherited POT or selected detector events.

An additive S2 response frontier must be causal as well as genealogical. A child
emitted before its parent's later escape is not downstream of that parent's
escape state; a naive ancestor-key veto would remove a legitimate exit. Conversely,
assigning a full continuation response to both an escaped state and a later
escaping descendant, or to each reentry, double counts. Until emission-versus-
crossing ordering, new-stage identities and exclusive downstream coverage are
established, retain the crossings but do not assign independent additive photon
responses. Charged escape and return-coupling remain explicit residuals. No S1
exit-flux or S2 closure is claimed by this geometry helper.

## Fail-closed acceptance

The initial positive resource ceilings are 100,000 steps/track, 1,000,000 SimParticles
and 1,000,000 photon-step records/event. These are caps, not asserted safe headroom.

- Require successful closed-file completion and saved event count equal to the
  declared source assignment; count zero-response events explicitly.
- Audit `StatusG4`, field-propagator/step-limit kills, SimParticle overflow,
  missing/unresolved Ptrs, missing collections and all truncation warnings.
- The photon recorder's overflow is warning-only and need not change `StatusG4`.
  A record count **at or above** its cap refuses completeness even without a warning.
- Sensitive-detector `g4.stepsSizeLimit` is a different control from the recorder
  cap; its preselection count can exceed the number of stored hits. Require its
  effective value zero, not merely a short stored collection.
- Retain problematic events in the ledger and fail the whole pilot's completeness
  claim; do not drop failed events and renormalize the survivors. An exception or
  timeout makes the affected run incomplete, regardless of a partially readable art.
- Check identity/parent links and conservation of attributed straw-gas total deposit
  and nonionizing deposit separately. Match the historical ionizing-energy objective
  only after its precise step/aggregation definition is audited.

A passing instrumentation pilot licenses analysis of that sample, not a new
campaign. Reserve development/held-out designs before selecting response variants.
All-point ranking, error and budget-classification gates remain separate.
