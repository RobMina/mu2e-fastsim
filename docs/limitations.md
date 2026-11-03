# Limitations

Per-stage lists: [CRV](https://github.com/RobMina/mu2e-crv-fastsim/blob/main/docs/limitations.md),
[tracker](https://github.com/RobMina/mu2e-tracker-fastsim/blob/main/docs/limitations.md),
[core](https://github.com/RobMina/mu2e-fastsim-core/blob/main/docs/limitations.md). This page has the project-wide
items and the beam stage.

## Project-wide

- **Scope.** The cosmic stages cover the extracted (KPP, no field) configuration only. Full-detector configurations
  with the field and the cosmic-induced fake-CE background are not simulated. Conversion electrons are carried from
  the stopping target to the tracker entry plane only (opt-in `beam/beamfastsim/cechain.py`, below).
- **Conversion-electron chain (opt-in).** Not modelled: foil and IPA support wires, OPA slots and supports, the TS4
  bend and the collimators upstream of COL5, field variation inside a layer, re-crossing of a surface within one
  10 mm helix step, exits through the z faces of the IPA end rings (the loss between the target exit and the tracker
  entry is 0.53 vs 0.49 MeV in Geant4 on average, with the same fractions above 0.5 and 1 MeV). Layers are flat in the
  local frame, with the IPA / OPA curvature as an equivalent field. Secondaries are not transported. The vacuum
  stepping runs in NumPy on the host.
- **Inputs are not published.** The chains read Mu2e-internal inputs; no collaborator release exists yet and its
  location is undecided ([setup/README.md](../setup/README.md)). The project image is not downloadable either.
- **Second account.** The quickstart has been run from an empty `HOME` with the same user, not yet from a second
  user account; access permissions of a shared release are therefore untested.
- **Device precision** is float32 by default in the CRV device kernels, float64 in beam, tracker and host code; the
  0.1 % pulse-PE offset is bounded (<= 0.3 %), not removed. Device output repeats run to run on one GPU kind with XLA
  autotuning off (the default) and is not bit-identical across GPU kinds.
- **Endpoints.** Fermilab EAF allows one MIG slice per user; the kit host must reach the Slurm login host by ssh; ESnet
  is parked (`core` branch `park/endpoint-esnet`).

## Beam stage

- Replaces the DS stage only (DS entrance to the stopping target): no TS transport, no flash `dts`, no IPA-stop
  output for downstream stages; it gives the stop fraction and stop positions.
- Physics: Geant4-matched mean loss and Urban fluctuations, Highland scattering, range-out and decay in flight;
  supports and absorbers are approximated. The stop fraction does not change with much finer Geant4 steps (1.0016 +-
  0.0020 nominal, 1.0019 +- 0.0011 moved), but the shape of the stopping edge does: finer steps raise the exit-energy
  variance at the edge by 15-40 %. The default transport has no comparison at that level, only at the stop fraction.
- The fast stop fraction is 0.5-0.7 % below Geant4 with paired errors on the same muons (0.9927 +- 0.0019 nominal,
  0.9934 +- 0.0011 moved against refined steps). Against refined steps the moved target is 2.15 sigma out with the
  independent-sample error, just outside its 2-sigma threshold. Any study needing the fraction better than 1 % should
  take this as a bias.
- The loss-model refinements on `park/beam-loss-studies` are not promoted: their slab comparison at the stopping edge
  is indeterminate against refined steps (variance ratios 0.89 / 1.02).
- Validated on the nominal 37-foil target and one 49-foil move; a broader Mu2eBO search space (per-foil holes, other
  materials) is not certified. Geometry changes outside the target-overlay keys are refused.
- The nominal Geant4 reference covers 105,350 muons of the 214,310 MuBeamCat events: an `NO_SUBRUN` input fault stops
  the stock job at event 105,100. A clean rerun to event 105,099 now exits 0 and agrees with it (1.0033 +- 0.0020).
- The beam stage needs JAX (no NumPy fallback).

## Studies kept off main

| Branch | Contents |
|---|---|
| `park/beam-loss-studies` | 39 beam loss diagnostics modules, their tests, closure/export/edge scripts |
| `park/flash-celeritas` | Celeritas image, adapter and probes (electron-flash feasibility) |
