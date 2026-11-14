# beam/: Geant4-free muon beam transport to the stopping target (prototype)

## Opt-in TS source comparison

The independent TS mu/pi route starts with [stock reference joins and paired
comparisons](docs/ts-reference.md); it does not change the default DS/CE chains.
See the [complete TS development guide](beamfastsim/tsREADME.md) for transport,
Run1A references, benchmarks, diagnostics and current limitations.

## Collaborator inputs

The umbrella [input inventory and setup](../setup/README.md) lists the MuBeamCat-derived
muon table, field map, geometry include tree and their publication gate.
No approved distributable beam input release is bundled. `beamfastsim.stage` can
fetch public map/geometry assets, but their hashes must be pinned alongside the
muon input. The streamed endpoint needs `FASTSIMKIT_FNAL_HOST` and reads `FASTSIMKIT_BEAM_DATA` (default
`$M2FS_INPUTS/beam_fastsim`) and `FASTSIMKIT_BEAM_CODE` (default `$FASTSIMKIT_SCRATCH/beam_code_kit`). The job
scripts take `M2FS_CODE`, `VENV` or `FASTSIMKIT_EAF_VENV` (a Python/JAX venv), `BEAMFS_DATA_ROOT` (the
`beam_fastsim` data area every closure output must be inside), `M2FS_DATA` and `PNFS_OUT` (grid staging) from your
site.env; each stops with a message naming a missing variable.

Package `beamfastsim` (imports `mu2efastsim` from `core/`; installable with `pip install ./beam`, metadata in
`beam/pyproject.toml`). Replaces the DS stage of the stock beam chain
(`Production/JobConfig/pileup/MuBeamResampler.fcl`) for the target-stop question: muons from the DS entrance (as
written by the POT stage, `MuBeamCat`) are carried through the field map, the stopping-target foils and the IPA in
batched JAX lanes (CPU or GPU) to their stop, decay or exit.

No Mu2e-internal data or results are stored here; the field maps, geometry and reference samples are read at FNAL.

| Path | What |
|---|---|
| `beamfastsim/fieldmap.py` | G4BL `.header/.bin` maps; trilinear lookup as Offline `BFGridMap` |
| `beamfastsim/simpleconfig.py`, `geometry.py` | Offline geometry text files -> foils, TSdA, IPA, outer PA |
| `beamfastsim/transport.py` | RK4 stepping, layer energy loss (G4 Urban via `mu2efastsim.eloss_jax`) and Highland scattering, range-out, decay, lane compaction |
| `beamfastsim/g4ref.py`, `compare.py` | per-muon G4 reference table and fast-vs-G4 metrics/plots |
| `beamfastsim/bench.py` | throughput on synthetic muons (no inputs needed) |
| `beamfastsim/rpcspec.py`, `rpccompare.py` | opt-in internal-RPC pair source from physical target-pion stops and source-kinematics comparison with saved stock RPCGun histograms; [physics, input scope and commands](docs/rpc-source.md) |
| `beamfastsim/cechain.py` | conversion electrons from target stops or given starts to the tracker entry plane: helix steps in the field maps, foils / IPA / OPA crossed with the core `eslab` or `eslab_jax` slab physics; per-electron plane state in an npz |
| `beamfastsim/cechain_jax.py` | the same chain as one device loop (`cechain --backend device`): slab steps of every layer type in one kernel (core `eslab_jax.lane_step`), helix steps with the maps on the device, surface crossings solved for the gathered lanes whose step reaches a surface, lane refill and compaction between launches; float64 or float32 |
| `beamfastsim/cesob.py` | sob stand-in of the stopping-target design studies: beam stops -> conversion electrons -> calorimeter-acceptance proxy (crystal map, disks' inner Al and carbon-fibre rings crossed with the slab physics) -> the studies' approximate sensitivity; the same estimate on a Geant4 EdepAna tree (`reduce`, `refsob`, bit-identical to the design studies), proxy checks against Geant4 (`calocheck`, `calodiag`), `hybrid` and `recalo`; fastsimkit task `beam_sob` |
| `beamfastsim/edepana.py` | opt-in EdepAna-format output: conversion electrons from stops through `cetracker.DeviceChain`, the calorimeter shower library at the disk fronts and the `cesob` wall proxy for side entries, the CeEndpoint storage filter, then `EDepAna/tree` with Offline EdepAna's branches, types and running `ngen`, the `edep` run record and summary block beside it, so the analysis server's `approx_ce_sensitivity` and `edep` readers run unchanged; fastsimkit tasks `beam_edep` (GPU) and `beam_edep_cpu` (one CPU core); [contract, coverage and validation](docs/edepana.md) |
| `beamfastsim/centuple/` | opt-in stock EventNtuple files from the fast CE chain: record (device chain with digis and the reco response, truth at the tracker surfaces, calorimeter clusters), device trigger, donor events from stock development files, and a PyROOT writer in the stock release (split TTree, stock layout), so the analysis server's EventNtuple analyses and Mu2eBO ce_chain run unchanged; fastsimkit task `beam_centuple`; [contract and coverage](docs/eventntuple.md) |
| `beamfastsim/diospec.py` | DIO electrons as the stock DIO primary production makes them: the tabulated spectrum restricted and sampled as Offline `BinnedSpectrum` / `DIOGenerator`, one electron per stopped mu- of each resampled stops event, isotropic, stop time plus the mu- lifetime in Al. The stops windows of stock files are recovered from their kept electrons (stops file events in tree-entry order, as the sequential `ResamplingMixer` reads them), so fast shards use the same stops events; starts in the `cechain --ref` format |
| `beamfastsim/cetracker.py` | opt-in hand-off from `cechain` to the field-on tracker (tracker `sim/fieldsteps`): the electron state at the tracker entry plane becomes a fieldsteps start with its event identity, then StrawGasSteps in the table layout of the art injector `StrawGasStepsFromTable`; first passage only, no secondaries (needs `tracker` on `PYTHONPATH`); `--starts FILE` takes DIO (`diospec starts`) or RPC (`rpcspec`) electrons instead of CE births, one event per row; the device chain (`run(positrons=)`, the bench entry `bench_run_starts`) runs RPC pairs, both legs, by default |
| `beamfastsim/diocompare.py` | fast DIO vs the stock DIO dts at the entry plane, paired per shard (same stops events, same generated electrons): yield per generated electron inside the filter-implying selection of `cecompare` and the ratio with the shard-residual error; momentum, cos theta, radius and time per generated electron with per-bin pulls and chi2; development and held-out shards; `--plot`, `--numbers` |
| `beamfastsim/dioreco.py` | DIO electrons through stock digitisation and LoopHelix reconstruction on the `cetracker` route: `single` drops the events with two primary electrons from the dumper tables (the one-primary hand-off and injector take one), `compare` pairs the fast and Geant4 arms per dts event (fit and CE-selection efficiency, selected tracks per 0.25 MeV/c and above 95-104 MeV/c, same-event momentum difference), `reduce` turns other EventNtuples (e.g. the production reco of the same events) into an arm, `plot` |
| `beamfastsim/cereference.py`, `cecompare.py` | Geant4 CeEndpoint reference table from `ReadVirtualDetector` trees; fast vs Geant4 at the entry plane (filter-implying selection, energy MPV/FWHM/tail fractions); `--mode pair`: two fast runs of the same starts (e.g. device vs NumPy) with paired errors |
| `fcl/g4ref_mubeam_ntpart.fcl`, `art/run_g4ref.sh` | stock DS stage of SimJob MDC2025ac with a per-muon tree |
| `art/run_fast_cpu.sh` | g4table + field check + fast run + comparison on a GPVM core |
| `art/eaf_beam.sh` | the same plus a benchmark on an EAF A100 slice (`core/eaf/eafhub.py run`) |

```bash
# GPVM: G4 reference (stock DS stage, ~12 ms/muon), then the fast arm on the same muons
bash beam/art/run_g4ref.sh $M2FS_DATA/beam_fastsim/g4ref_full 214310 1
bash beam/art/run_fast_cpu.sh $M2FS_DATA/beam_fastsim/g4ref_full $M2FS_DATA/beam_fastsim/fast_v1_20k 20000
# conversion electrons: starts of a Geant4 reference table, or isotropic CE from a stop table
PYTHONPATH=core:beam python3 -m beamfastsim.cechain --offline-root $OFFLINE --maps DSMap.header TSdMap.header \
  --stops g4ref.npz --n 8000 --backend jax --out ce_fast.npz
# the same on a GPU in one device loop (float32; lanes refilled from the starts)
PYTHONPATH=core:beam python3 -m beamfastsim.cechain --offline-root $OFFLINE --maps DSMap.header TSdMap.header \
  --stops g4ref.npz --n 400000 --backend device --device-float float32 --lanes 131072 --shrink 32 --unroll \
  --out ce_fast_dev.npz
PYTHONPATH=core:beam python3 -m beamfastsim.cecompare --ref ce_g4ref.npz --fast "ce_fast*.npz" --mode stops \
  --offline-root $OFFLINE --maps DSMap.header TSdMap.header --out compare.json
# DIO: stop table (ReadVirtualDetector tree of the stops file, events in tree-entry order), starts paired with the
# stock reference files (cereference --code 166), the device chain, then the per-shard comparison
PYTHONPATH=core:beam python3 -m beamfastsim.diospec stops stops_ntpart.root --art stops.art --out stops.npz
PYTHONPATH=core:beam python3 -m beamfastsim.cereference nts.readvddio.*.root --code 166 --gencount 4000 --out dio_ref.npz
PYTHONPATH=core:beam python3 -m beamfastsim.diospec starts --stops stops.npz --ref dio_ref.npz \
  --table $OFFLINE/Offline/EventGenerator/data/heeck_finer_binning_2016_szafron.tbl --out dio_starts.npz
PYTHONPATH=core:beam python3 -m beamfastsim.cechain --offline-root $OFFLINE --maps DSMap.header TSdMap.header \
  --ref dio_starts.npz --backend device --device-float float32 --lanes 131072 --shrink 32 --unroll --out dio_fast.npz
PYTHONPATH=core:beam python3 -m beamfastsim.diocompare --ref dio_ref.npz --starts dio_starts.npz --fast "dio_fast*.npz" \
  --offline-root $OFFLINE --maps DSMap.header TSdMap.header --out dio_compare.json
PYTHONPATH=core:beam python3 -m beamfastsim.diocompare --plot dio_compare.json dio_vd13.png --numbers dio_compare.json numbers.json
# sob stand-in for one target geometry: stops, CE, calorimeter proxy and the approximate sensitivity
PYTHONPATH=core:beam python3 -m beamfastsim.cesob point --g4 g4ref_run1baa_vd8.npz --z-up 3900 --geom overlay.txt \
  --electrons 100000 --procs 32 --out sob_point
# any machine: unit tests and benchmark
PYTHONPATH=core:beam python3 -m pytest -q beam/tests
PYTHONPATH=core:beam python3 -m beamfastsim.bench --n 2000
```

## CE tracker hand-off

`beamfastsim.cetracker` uses the tracker's default `steps.STRAW_GAS` material (`state="gas"`, the Geant4 gas
density-effect branch), with the electron loss model. There is no gas-state selector in its Python or CLI API.
The hand-off remains opt-in, first passage only, without secondaries; it needs `tracker` on `PYTHONPATH`.

## Persisted reference audit

Audit existing outputs before reusing a partial reference; this does not run transport or require JAX/CVMFS:

```bash
PYTHONPATH=core:beam python3 -m beamfastsim reference-audit \
  --nts "$REF/nts.owner.beamfs_g4ref.version.sequencer.root" --g4 "$REF/g4ref.npz" \
  --artifact "$REF/job.fcl" --artifact "$REF/job_config.txt" \
  --artifact "$REF/job.log" --artifact "$REF/mubeamcat.txt" \
  --out "$REF/reference_audit_UNIQUE_TAG.json"
```

The audit selects the highest `mutree/ntpart` cycle, records every available cycle's entry count, counts distinct
persisted event/primary IDs, hashes the ROOT/table/configuration artifacts, and compares all raw primary columns to
the cached table in row order. Duplicate primary IDs or a table mismatch make the CLI exit nonzero.
Add `--geom PATH --offline-root ROOT` to recompute `q`, `at_rest` and `foil` from ROOT rows, compare them to the
cache, and record the effective geometry plus SHA-256/size/path of every file traversed by the beam geometry reader.
`--baseline-geom Offline/Mu2eG4/geom/geom_common.txt` additionally checks the parsed overlay contract below.
These are current-reader checks, not historical G4 binary/geometry certification. Empty-muon events and source art
contents are not certified. Review the saved job outcome separately; particle rows are not event counts.

`transport_all` enforces an exact positive-integer step cap, including the final partial chunk. Foil crossings own
the arrival endpoint and exclude departure in either direction. `make_stepper` rejects nonfinite/malformed foil
arrays, invalid radii/thicknesses, overlapping or unordered foils, and centre spacing smaller than `ds`. Treat the
geometry arrays as immutable after compilation. This is a stepping precondition, not a full geometry contract:
the inherited material, support and absorber approximations are not certified by this check.

`run` and therefore `point` reject effective parsed changes outside these target-overlay keys, relative to
`Offline/Mu2eG4/geom/geom_common.txt` in the selected Offline root:

- `stoppingTarget.{radii,halfThicknesses,zVars,deltaZ,z0InMu2e,holeRadius,holeRadii}`
- `protonabsorber.distFromTargetEnd`
- `hasTSdA`, only set to `false`: the TSdA absorber is then not built (the Mu2eBO foils studies run Geant4 without
  it); the default geometry keeps it

This includes changes through nested includes, not just the top-level file. A scalar hole change shadowed by a
per-foil `holeRadii` vector is rejected. The baseline's unmodelled supports/absorbers remain approximations; parsed
material, tilt, field-axis and unrelated detector changes are rejected rather than silently scored. This guard
covers effective **parsed** keys only: the minimal SimpleConfig reader is not a general syntax validator, and a
user-selected Offline root is not an immutable approved baseline. Full G4 support for per-foil holes is release
dependent. Allowed target changes still need paired physics validation; they are not a certified BO search space.

For a foil stop, time and proper time use the crossing fraction of the pre-step flight, consistent with the saved
thin-plane stop position. No within-foil slowing time is integrated; surviving crossings, IPA stops, boundary/decay
ordering, grazing paths and the low-energy model remain approximations requiring physics convergence checks.
A shared seed is not layout-invariant: chunking and compaction affect random draws.

## Layer physics

Layers lose energy in slices with the local restricted dE/dx (fast Bethe law, Urban fluctuations) and scatter with
one Highland kick. The range-integrated alternative with Geant4 11.3.2 aluminium tables (`--physics g4`) and the
loss-model studies behind it (numerical convergence, finer slicing, frozen-entry loss histories, single-foil closure
against Geant4, slab and edge diagnostics) are kept on branch `park/beam-loss-studies`. Refining the layer loss model
did not improve agreement at the foil edges.
