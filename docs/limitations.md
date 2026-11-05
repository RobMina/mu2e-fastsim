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
  10 mm helix step, exits through the z faces of the IPA end rings; the end rings sit radially against the IPA, and a track leaving one starts its next step on the other's face, where the strict crossing test misses that shell about half the time (both backends) (the loss between the target exit and the tracker
  entry is 0.53 vs 0.49 MeV in Geant4 on average, with the same fractions above 0.5 and 1 MeV). Layers are flat in the
  local frame, with the IPA / OPA curvature as an equivalent field. Secondaries are not transported. The vacuum
  stepping runs in NumPy on the host.
- **sob stand-in (opt-in `cesob`).** The calorimeter selection is a proxy: a helix path of at least 17.8 mm inside
  crystals, with the disks' inner Al and carbon-fibre rings crossed with the slab physics. Not modelled: tracker
  material, the calorimeter front and back plates, the front-end electronics boxes and pipes (electrons that graze
  the first disk's inner edge and cross them are kept too often: +1.6 % in acceptance against Geant4), and showers.
  The fast chain uses EM option 4; full-simulation design points on FTFP_BERT (option 0) lose slightly less energy.
- **Trigger replay (opt-in diagnostic).** Complete-history CE stock control has zero accepted/state/`wasrun`/
  mapped-endpoint differences across 52 physics paths and 28,044 development events;
  a frozen independent CE job also passes all 52 paths over 28,148 events without tuning. The complete
  28,044-event tracker-fast/stock-calo hybrid is measured but not physically closed: representative CPR CE
  acceptance is 5.805 percentage points lower, with upstream reach differences. Missing secondaries/returns,
  separate tracker-electronics RNG and retained original G4 calorimeter history are not separated causes.
  Two BO-final selections have zero eager/JIT measured-array mismatches against the hybrid, retaining actual
  upstream fast reach; this is not upstream reconstruction/full-trigger emulation. Independent fast confirmation,
  numerical boundaries, independent feature extraction and dominant-cost acceleration remain open, as do
  cosmic-era control and fast-calorimeter arms. The quiet pair shows no whole-job acceleration.
  R7 is paused by user as of 2026-11-05 06:10 EST, with no R7 jobs running. Only the first
  full-stock imported-SGS C parent is complete (2,811 events, 138,045 SGS rows, 11 zero-SGS
  events): ordered SGS/generator numerics and retained original serialized products/statuses/SGS
  pass, but imported particle state is not lossless and no electronics-only or physics closure
  follows. Nine parents and the full-history C trigger remain unlaunched. Resume requires
  [explicit authorization and admission gates](../trigger/PAUSE_RESUME.md).
  All replay raw indices/packed words differ after deliberate path shortening; full-art serialization identity
  is false. ROOT IDs and ordered schedule registry bindings are independently checked, not recursive module
  configuration resolution. Original selected production digis omit counter-advancing events: the earlier
  21,078-event selected control has four lumi/phi differences despite the other 50 paths matching. Recovering
  complete authentic source history resolves those differences, not event-number modulo or stock-bit copying.
  Production outputs are trigger OR truth-selected, not an unbiased generated-event denominator. Same path
  names/prescales across releases do not imply equal hit conversion or fit settings.
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
