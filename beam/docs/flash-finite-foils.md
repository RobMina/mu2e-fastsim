# Finite-foil transport extension

## Required physics change

The current flash navigator admits arbitrary coaxial annular foil arrays but
transports an in-material charged track through a laterally infinite slab. A
photon-birth cut outside the annulus does not repair the parent's excess loss,
scattering or downstream photon production. Finite charged transport must be
implemented before claiming general-design flash accuracy.

The core per-leg hook and a restricted annular geometry primitive are implemented
as development components, **not connected to the source navigator**. Finite
charged transport is therefore not an available source-model capability yet.
Shared CE defaults and field maps remain unchanged. The CE and flash device
navigators use common target-geometry helpers, but do not enable this boundary
model; see [shared-transport.md](shared-transport.md) for ownership and interfaces.

## Implemented components

`lane_step(..., boundary=..., safety=...)` and the recording variant
`lane_step_photons` accept paired deterministic callbacks. The boundary callback
replaces the axial query on each actual MSC leg; safety includes lateral surfaces
in the range-trapping decision. Boundary
exits flush pending loss on the actual path and suppress endpoint hard processes.
The random-key split, draw shapes, recording capacity and default path are
unchanged. Core tests cover forced first/second-leg exits, both charges, field and
straight motion, loss accounting, inactive lanes and stopping precedence.

Core `mu2efastsim.geometry.annulus_jax.crossing` (also re-exported by
`flash_annulus_jax`) provides vectorized float64 exit/entry queries for
straight motion and **axial fields only**. It solves plane intersections and both
radial helix phases analytically, including full-turn leave/return and radial
entry after axial admission. It handles holes, no-hole foils, tangencies and
boundary starts without nudging; `safety` includes faces, rim and hole. Surface
codes match `flash_exit`. A supported no-hit returns infinity; unsupported or
numerically unresolved input returns NaN and `supported=False`. Tilted fields,
extreme phase/curvature and nearly repeated roots within the roundoff bound are
not accepted as clear paths. This is a restricted development solver, not a
validated arbitrary-field or arbitrary-coordinate-range algorithm.

Geometry tests compare independent straight-ray and circle-intersection formulas,
with regressions for false tangent transitions and long-path root roundoff.
Physical Al-table integration tests exercise real truncated core steps; a
mean-loss component test follows a hole exit, vacuum crossing and opposite-wall
re-entry at its actual depth. These do not test source-navigator integration or
agreement with G4. Current CPU regression checks pass 52 core geometry tests,
nine beam/core integration tests, 66 core-hook tests and 26 photon tests; a
separate namespace test verifies that importing the geometry package does not
load JAX. The navigator still needs per-foil identity,
lateral entry handoff, conservative sparse-vacuum candidates and a general
tilted-field solver. An unresolved geometry result must stop or refine the
geometry decision before physics is committed; passing its NaN distance directly
to the core callback would violate the callback contract.

## State and core interface

The authoritative in-layer state is `st['slab']`, not the navigator's last vacuum
position. Its local position/direction, energy, pending continuous-loss path and
accumulated path are transformed through the saved entry frame. Retain the actual
foil index separately from the material/thickness type: several foils can share a
material type but have different radii or holes.

A material step may contain two straight/helical legs separated by a sampled
multiple-scattering direction. A scalar distance computed only before that step
cannot safely limit both legs. Use the core's optional deterministic
per-leg boundary query, evaluated at the actual leg start/direction/momentum and
proposed length. Compete lateral exits with existing axial faces before committing
transport or material effects. Near-boundary trapping/safety must also consider
the rim and hole, not just planar faces.

For a truncated leg, apply continuous loss to the actual traveled/pending path,
suppress hard interactions beyond the boundary, preserve stopped-versus-exited
precedence and emit no beyond-boundary photon. Keep full material batches, record
capacity and RNG draw shapes. Geometry consumes no random numbers; do not retry
stochastic physics or reset the slab clock to repair a geometry overshoot.

## Boundary and handoff requirements

Straight legs use finite cylinder/plane intersections. Curved legs need earliest
root isolation on the actual frozen-field helix, including turning points and
leave-and-return trajectories. Endpoint sign tests alone are insufficient.
Boundary-start classification depends on direction/curvature; tolerances control
classification and root accuracy, not position nudges.

A lateral exit uses the normal finished-layer projection and adds path/time once.
Vacuum navigation must then recognize rim/hole re-entry as well as axial faces.
For example, a track can cross a central hole and re-enter the opposite wall even
without a field. Entry through a cylindrical wall begins at its actual nonzero
axial depth. Sparse CPU vacuum bursts must stop before these candidates just as
they do before axial crossings.

## Minimum validation

- Disabled-mode legacy state/record parity; enabled trajectories far from edges
  retain the existing transport result.
- Analytic face/rim/hole competition, no-hole foil, tangent and exact boundary
  starts, zero transverse velocity, curling exit/re-entry and both charges.
- Forced MSC exits on each leg separately; near-rim safety, pending-loss flush,
  energy balance, path/time and no beyond-boundary emission.
- Rim/hole vacuum re-entry at nonzero depth; no duplicate material/path/time;
  capped partial-layer state and chunked/hybrid scheduling consistency.
- Matched small G4 cases for charged first exits followed by material-enabled
  loss/brems distributions. Existing inclusive references record photon steps,
  so they alone do not directly validate charged boundary paths.
- Separately compare primary-brems first-foil exits on the existing ordered DS
  input sample; do not conflate this with full shower or tracker-response closure.
