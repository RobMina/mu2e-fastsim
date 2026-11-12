# Component-training capture pilot

Status: diagnostic capture of complete local interactions in finite components (simple boxes and the v10 R1 CRV
module 0). Every record carries `bankAdmissible=false`; no training bank exists and no production chain reads these
records. Existing physics banks are unchanged. Passing every checker below is necessary for a future bank, never
sufficient.

## Integration

The reference Mu2eG4 cut factory is a closed set of built-in predicates. `type: tool` delegates to an art class tool
implementing `IMu2eG4Cut`, with a nested `tool` parameter table. An independent optional `IMu2eG4Observer` interface
allows the **top-level common cut** to receive classification and pre/post-tracking callbacks. Built-in cuts do not
implement this interface. It does not alter the base cut interface/vtable or change the built-in cut decisions.

Use one recorder as the top-level `Mu2eG4CommonCut`, with empty stacking-only/stepping-only cuts. Do not bury it inside
union/intersection: their short-circuit semantics do not forward lifecycle observations. A recorder must implement the
complete event lifecycle, including cleanup of internally rejected events. Constructors must not book ROOT trees:
declaration-only temporary cut instances are also constructed. Prefer event products and an analyzer.

`classifiedTrack(track, killed)` runs after the stacking decision, including when the stacking-only predicate rejects
the track. Classification can occur more than once; IDs may not yet be final for rejected tracks. Do not treat this as
a unique birth census or manufacture art pointers for tracks never tracked. `beginTrack` follows SimParticle-start
creation; `endTrack` runs before SimParticle-end serialization and sees the actual G4Track state. Suspension and
postponement are not terminal deaths. Preserve real G4 information before a first-exit artificial kill changes it.

## Recorder contract

The schema-1 recorder writes five tables (events, tracks, classifications, steps, volumes) and a links product: the
creator step of every secondary (matched by pointer from `G4Step::GetSecondaryInCurrentStep()`), whether it was born on
the envelope surface, and whether it exits at birth. Each event carries a manifest (Geant4 version, physics list,
process lists, declared Geant4 behaviours) and its FNV-1a 64-bit hash.

- Every in-component G4 step: pre/post positions, momenta, total/kinetic energy and mass, global/proper times,
  raw/nonionizing energy deposit, length, material/placement identity, defining process and step status, including
  zero-deposit steps.
- Full integer source/event/track/parent identities.
- First boundary crossing: exact post-point state (its direction may point back in after multiple scattering), distinct
  exit reason, then stop only that branch.
  Legitimate already-produced children are not killed because their parent exits.
- Boundary-born secondaries follow one ownership rule: owned by the component, an exit at birth (killed at stacking
  with its own reason), or an error.
- Natural terminal states, no-step tracks and artificial/resource kills are classified separately. A track stopped by
  its own energy loss on a boundary-limited step is a natural stop. At-rest physics remains enabled.
- Mu2eG4 counts such a stop as a field-propagator kill (`StatusG4` status 1) unless the track is an electron or a
  proton. The dump and `capture.audit` both recompute those natural stops (`nNaturalBoundaryStops`) and accept
  status 1 only when every counted kill is one; any other field-propagator kill invalidates the event.
- Any overflow, missing ancestry, unsupported classification or resource kill invalidates the trial; it is never
  silently truncated or turned into a physical no-response outcome. A rejection stops the sample; surviving trials
  are never selected.
- No detector selection, positive-deposit threshold, neutrals filter or ancestor-only particle compression.

The boundary is a real finite G4 envelope; a coordinate-plane predicate alone can overshoot it. Membership includes
daughters and uses physical placement/touchable ancestry, not only logical-volume names. The record includes passive
components and interior air/gaps, not just scintillator. Outside transport is not part of the trial.

## Checkers

All are vectorized NumPy/pandas readers of the dumped ROOT tables; none imports a production chain.

| Module | Checks |
|---|---|
| `capture` | strict audit: event counts, identities, ancestry, boundaries, rigid transforms; links and manifest hashes |
| `capture_boundary` | the five constructed boundary-born ownership cases |
| `capture_accounting` | per-step and per-event energy accounting (below) |
| `capture_partition` | one recorded component beside an unrecorded touching neighbour: one transport owner per region |
| `capture_compare` | small-sample response distributions; `--exact` compares two captures field by field |
| `capture_inventory` | geometry inventory and primary traversal |
| `capture_lifecycle`, `capture_callbacks` | negative lifecycle fixtures and constructed callback sequences |
| `capture_zero_step`, `capture_secondary_zero_step`, `capture_end_kinematics` | zero-step and endpoint fixtures |

`capture_compare --exact` treats `Transportation` and `CoupledTransportation` as one process class and compares
strings by value; CPU and real-time fields are excluded. A distribution comparison is a diagnostic, never an
equivalence certificate.

### Energy accounting

Energies are total energies, so rest masses are explicit. For each step inside the component,

    R = E_pre - E_post(kept) - deposit - sum(E_birth of the step's children) + medium

where `E_post` is dropped when the step destroys the particle and the medium term is `m_e` times the net electron
number the step takes from the material, plus the rest mass of any recoil nucleus an EM step sets in motion
(Geant4 labels a hydrogen recoil as a proton).
Transport and EM steps must close within `rtol = 1e-9` (relative to the pre-step energy) plus `atol = 1e-9` MeV.
Hadronic steps report an explicit nuclear term and are never forced to zero. Each event must close within `rtol`
plus `atol` per step, counting deposits, first exits, exits at birth, energy left inside and every declared term.

Decay steps and a few EM steps carry declared terms, each Geant4 behaviour rather than a tuned value, each gated step
by step at the same tolerance and counted in the report:

- `G4MuonDecayChannel`: daughters total `m_mu - m_e` in the muon rest frame. The gate is Lorentz invariant: the
  missing 4-momentum `D = P_mu - sum p` has rest-frame energy `D . P_mu / m_mu = m_e` (`Q = m_e` at rest; larger in
  flight).
- A nucleus without a decay channel (e.g. a triton with radioactive decay off) is killed at rest by `Decay` with no
  products: `Q = 0`, its rest mass ends in the material.
- A photon made in a boosted decay carries a rounding-level dynamic mass (`m <= 1e-6 E`); the EM process that absorbs
  it shares only its kinetic energy, so its pre-step mass is that step's declared term.
- A positron made in a boosted decay keeps its rounding-level mass offset from `m_e` (within the e+- mass bound
  below); annihilation makes its photons from `m_e`, so the offset `m - m_e` is that step's declared term.
- `G4Decay` makes daughters at the parent's dynamic mass `m` and boosts them with `gamma = E/M`, so a decay may give
  `Q = E (1 - m/M)` for `|m - M| <= 1e-6 M` (`M` is the most common mass of that PDG in the file).

Every other decay must conserve energy. Electron and positron masses must agree within 1e-6 of their median. A decay
product's dynamic mass is rebuilt from `E` and `p`, so its rounding grows as `eps E^2 / m`: its bound adds 16 ulps of
`E^2` in `m^2` (5e-6 MeV for a 38 GeV positron from a boosted pi0 Dalitz decay).

## Geometry and validation order

The bounded construction and remaining gates are in [canonical CRV component](crv-component-construction.md).

1. **Recorder smoke geometry:** simple finite box in a vacuum world to test exact entry/exit, backscatter, stopping,
   neutral transmission and empty response. This is a software fixture, not a CRV module bank.
2. **Canonical CRV module:** reproduce a specific module from the pinned Offline construction, including all layers,
   absorbers, strongback and cover. Verify placement/material inventory and local sensitive-ID map against the source.
   Do not substitute a uniform slab or the union of scintillator bounds. `constructCRV` builds whole configured sectors
   and some passive structures outside their mother; copying only that mother is not sufficient.
3. **Geometry covariance:** translated/rotated versions, side orientations included, under the same local field-free
   conditions. Compare complete response distributions using independent trials; exact-state replay is a software test.
4. **Neighbor test:** explicit transport of recorded exits to another component, including returns as new interactions.
   Compare with full G4 and test for double counting.

The small validation matrix is in [`training/component-pilot.json`](../training/component-pilot.json). Repeated fixed
states validate the current exact-state library API. A continuous empirical conditioning law and broad physics
coverage remain separate work, not something this pilot promises to establish.

## Batch execution gate

Use CPU batch jobs, one G4 process/thread per job; no GPU request. Start with a completed capture smoke test, then the
bounded matrix. Measure actual wall time, peak memory and bytes/event from finished jobs before choosing shard sizes;
do not extrapolate total runtime from the smoke test. Stop the matrix at the first failed audit or accounting gate.

Before dispatch: recorder build verified; capture/geometry checks passed; runtime image and custom libraries staged
with checksums and compared bit for bit against a reference smoke; output paths unique; seed/run/subrun mapping
recorded; no existing job with the same request. Worker scripts never build software.
