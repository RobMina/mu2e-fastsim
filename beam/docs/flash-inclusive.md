# Inclusive photon reference

## Purpose

Measure the photon exposure missing from detector-selected ancestry, then test
source, response and end-to-end prediction independently. The reference retains
zero-response photons and zero-photon/zero-deposit input events. It does not fit
an overall normalization to the observed design ratio.

`fcl/flash_inclusive.fcl` is an opt-in direct continuation of existing
`EleBeamCat.MDC2025ab` DS input. It uses stock Mu2eG4 and the previously built
read-only `FlashReferenceExporter` analyzer. No transport library or shared
configuration is changed. The initial two-design run checks instrumentation;
it is not an absolute prediction of the historical board sample.

## Initial configuration

- Same ordered input events for both designs, all `BeamFilter:Beam` states.
- Existing Offline v13_41_01 / p109, FTFP_BERT, stock `bfgeom_v01` field.
- Explicit frozen effective geometry for each design, not a moving geometry alias.
- Range production cut 0.010 mm; trajectories disabled; no stacking/common cuts.
- Photon-only, non-killing all-step recorder, including zero-edep steps.
- Raw tracker, calorimeter, CRV and virtual-detector products retained.
- Unconditional art output and flat event/particle/step/status/volume tables.
- No resampling, added pulse-time offset, source-time cut or detector selection.

The runner supplies event count, skip, seed, geometry and output names. Its
`expectedEvents` exporter setting must equal the actual requested input count.
No automatic large-campaign submission follows the pilot.

The geometry sets `g4.stepsSizeLimit = 0` (no sensitive-detector hit limit).
The photon recorder and SimParticle collections each have a one-million-row
per-event ceiling; the track step ceiling is 100,000. Status failures, field or
step kills, pointer failures and collection-limit warnings remain visible.
A truncated event is not a successful zero-response event and is not removed
from the denominator to obtain a better result.

## Boundaries and response requirements

Keep two spatial observables distinct:

1. First outward exit from the photon's birth foil: both axial faces, outer rim
   and inner hole. This is the interface closest to the current fast S1.
2. Outward crossings of a closed envelope enclosing the complete target,
   including its lateral surface and endcaps. Two transverse virtual detectors
   alone do not define this envelope.

The all-photon step record supplies pre/post positions and momenta, pre-step
GlobalTime, step length and end process. These support reconstructed spatial
crossings, not native G4 boundary tuples. It has no post-step time or unique
post-volume identifier. Photon-cut coordinates are Mu2e coordinates; raw straw
positions are tracker coordinates. Birth volume comes from SimParticle stage
and start-volume index joined to the SubRun volume table, not the step copy number.

`flash_g4_photons` retains every identified foil birth and reconstructs its first
spatial exit from the stored segments. Near-boundary discrepancies are flagged,
not rounded into a crossing. An immediately following segment can resolve one
only when its identity, endpoint continuity, energy and direction agree and its
endpoint lies clearly beyond the same face. The combined chord intersects the
unchanged foil geometry; saved coordinates are never nudged. Such results carry
`two_segment_reconstruction` and retain the original step-index/ambiguity fields.
Unresolved photons remain in the output and must not be treated as absorbed.

Count only newly simulated photons, not copied upstream ancestors. Retain birth
process and parent identity to separate primary brems from annihilation,
secondary brems and other photon production. Publish all-photon and matched
primary-brems comparisons separately.

For additive response, each tracker deposit must have at most one owning
interface photon. A photon and its later descendants cannot both own the same
deposit. Reentries and descendants born before an ancestor crosses require an
explicit causal assignment; a nearest-photon birth attribution is not by itself
a boundary-response definition. Start from the complete interface exposure and
left-join attributed deposits, filling absent contributions with zero. Keep
unassigned/non-target energy as an explicit residual.

Measure energy-only response as a diagnostic, then test transfer between designs
and variation with position, direction and time. The basic validation sequence
is: fast flux versus G4 flux; response model supplied with G4 photons versus G4
deposits; fast flux times response versus the complete G4 result.

## Matched S1 component check

`flash_g4_primary` selects eBrem photons whose direct parent is a new-stage
input electron/positron continuation. It does not follow a new-stage secondary
back to a primary and re-label its brems as primary emission. Uncollided exit
selection checks recorded process history before the crossing, not rounded
energy equality; a two-segment reconstruction includes its transport anchor in
that history. Missing/ambiguous records are explicitly unresolved.

`flash_flux_compare` compares saved fast primary-brems flux for paired designs,
requiring identical represented event identities and a shared declared source-event
denominator. Extra denominator events must be known shared zero-input events, not
inferred missing records. It reports count and energy-sum ratios with paired-event
and SubRun-cluster uncertainties; few-cluster errors remain fragile. Spectral
shape diagnostics account for the fitted observed count ratio and event covariance
when the photon-to-source mapping is available; without it, shape uncertainty is
not evaluated. That fitted ratio is a shape diagnostic, not a detector-response
normalization. Spectra cover the declared bins, while count ratios use all energies.
Dropped births or capped parents are rejected; path-limit loopers still make the
result a truncated-source diagnostic. Unsupported attenuation remains a caveat.
This comparison is primary-only, not all-photon, full-shower or detector closure;
a historical detector-deposit ratio is not a flux closure target.

`flash_s1_reference` compares this component with saved fast uncollided first-foil
weights on the same complete input-event universe, combining disjoint saved G4
shards without rerunning transport. Fast event totals obtain that universe from
the extractor's `*_events.npz` ledger: `flash_flux --states` discovers its sidecar,
or `--event-ledger` supplies one explicitly (also for old `--births` files).
New births retain a supplied ledger for later postprocessing. Unique exact integer
identities, source-row membership and equality to the scalar source-event count
are required. Only these verified identities are zero-filled when no e-/e+ source
rows exist; missing/unverified inputs still fail the matched comparison. See
[flash-flux.md](flash-flux.md#time-and-normalization) for the saved fields.
Count and energy-sum comparisons have paired event and SubRun uncertainty
diagnostics. Agreement of counts alone does not
establish spectral or energy closure. Fast path-limit tracks and unresolved G4
exits remain caveats; neither is removed from the event denominator.

The charged finite-annulus extension needed for general-design validity is
specified separately in [flash-finite-foils.md](flash-finite-foils.md); it is not
implemented by these reference-analysis modules.

## Timing and normalization

The [historical board observable](flash-board-observable.md) defines the exact
7.506758e-6 MeV/generated-resampler-event budget, saved filter/aggregation
settings, source/pulse policy and independent two-anchor normalization checks.
Those checks establish the comparison quantity, not fast/G4 closure.

Raw times and deposits are retained. The stock main-flash event filter uses
strict `fmod(t,1695) > 250 || fmod(t,1695) < 10`, together with detector energy,
particle and multiplicity conditions. Applying this predicate to individual raw
steps is a timing diagnostic, not a reproduction of the aggregated event filter.
The historical board files are **EarlyEleBeamFlash**, with prescale one and
without the main-flash time cut. Board closure must reproduce that actual
objective, rather than silently substituting time-selected raw tracker energy.

The direct DS pilot is normalized per processed input event and per input state.
It does not supply POT normalization. Historical board production used
`EleBeamCat.Run1Baa` TS3 input with a resampler and proton-time offsets; its
11 million GenEventCount entries per design count downstream generated draws.
The source pool, offsets, aggregation, selection and geometry must be reconciled
before comparison to the board budget. Direct DS/G4 agreement alone cannot do so.
The board configuration also kills electrons/positrons below 1 MeV in named
upstream/vacuum volumes and suppresses neutrinos at stacking. The initial DS
pilot intentionally has neither cut. A board-compatible arm must restore the
board transport settings; changing recording inclusivity must not silently
change the physics being compared.

## TS3 source check

`fcl/flash_inclusive_ts3.fcl` derives from the DS recording configuration and
restores the saved board's low-energy electron/positron and neutrino cuts and
sensitive-volume list. It reads the exact board input pool directly, retaining
all events. This isolates source/transport differences without requiring the
resampler's event-level volume metadata. It is still not the complete board
chain: proton-time offsets, downstream step aggregation and event selection are
not applied. The flat export preserves the raw information for those tests.

## Generalization requirement

The 66 historical board points are a benchmark, not the supported design space.
Inference must accept a new geometry without a board-point identifier, a
per-design fitted normalization, or a new Geant4 reference run. Geometry and
material parameters drive transport; the historical points do not define a
lookup table. The current primary-brems/planar-layer implementation has not yet
established this capability for the full flash observable.

Declare the supported geometry, material, source and field domain explicitly.
Unseen designs within that domain must be supported; unsupported materials,
shapes or parameter ranges must be reported, not silently mapped to a known
point. Arbitrary new foil designs do not imply arbitrary detector geometries or
materials are already implemented.

At first emitting-foil exit, the remaining foils and absorbers are part of the
response transport. An energy-only response averaged over historical points
cannot be assumed universal. Either transport the remaining variable geometry
explicitly, or validate a response conditioned on photon energy, position,
direction and time plus the relevant geometry. A shared closed interface outside
all variable material may make a reusable external-detector response possible,
but only with fixed external geometry and tested treatment of particles returning
into the target. Photon-only coverage, charged-particle contributions and
non-target deposits need separate closure. Event selection must be reproduced at
event level where it prevents additive per-photon factorization.

### Scoped geometry corrections

The flash CLIs use `flash_geometry.from_offline`; shared beam/CE loaders are not
changed. This reader repeats the final short `zVars`/half-thickness entry as
Offline does, places the IPA using its absolute pin or the target mother end plus
VD clearance and the relative distance, and derives TS5 from `ts.rTorus` rather
than TSdA. The flash navigator removes the TSdA absorbing plane when it is absent.
It rejects noncoaxial or tilted foils, non-Al overrides, invalid/overlapping foil
arrays, unsupported mother-cylinder extents and out-of-range IPA distances.
The radial containment restriction is conservative for conical target mothers;
it is not full GDML overlap certification. Global material selection, complete
SimpleConfig syntax and arbitrary changes outside this represented subset still
require a stronger admission contract.

These corrections change the fast model. Earlier source and throughput results
remain evidence for their frozen code/geometry only; they are not re-labelled as
results of the corrected implementation. No new G4 transport is needed merely to
reanalyze saved records, but matched fast-source validation must use the corrected
geometry before assessing transfer.

## Scaling and validation

First inspect the two-anchor pilot's complete input counts, inclusive photon
records, empty events, status and raw detector sums. Then accumulate a useful
anchor sample with frozen settings. Preserve a separate held-out event set;
event holdout alone does not test generalization to unseen designs.

Before response fitting, define development and design-held-out samples, recording
which historical results have already informed development. Supplement the board
points with new physically valid geometries covering the declared parameter
space, including sparse regions, parameter extremes and unfamiliar combinations
of foil radii, thicknesses, holes, spacing and longitudinal placement. Include
foil-count changes if they are within the supported domain. Freeze the model
before evaluating the reserved new designs; run Geant4 for validation, not as an
inference prerequisite for each new design. Failures drive documented domain or
physics changes followed by a fresh independent test.

After the interface and normalization are exercised, expand references across
the board points and new designs with compact photon exposure and event-level
response outputs. Do not retain all-step raw art for every large production shard
merely because compute is available. Use measured output sizes and completed-job
runtimes for resource choices, not linear runtime extrapolation from a tiny test.

Report source, response and end-to-end closure separately on unseen geometries,
including absolute energy-deposit errors and uncertainties, spectral comparisons,
and failures versus geometry parameters. The original 66-point metrics remain
Spearman rank at least 0.8 and agreement with the 7.507e-6 budget decision of at
least 95%; meeting them is not sufficient evidence of generalization. Response
transfer and the missing non-target contribution are physics tests, not assumptions
implied by those ranking thresholds.
