# beam/: Geant4-free muon beam transport to the stopping target (prototype)

Package `beamfastsim` (imports `mu2efastsim` from `core/`). Replaces the DS stage of the stock beam chain
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
| `fcl/g4ref_mubeam_ntpart.fcl`, `art/run_g4ref.sh` | stock DS stage of SimJob MDC2025ac with a per-muon tree |
| `art/run_fast_cpu.sh` | g4table + field check + fast run + comparison on a GPVM core |
| `art/eaf_beam.sh` | the same plus a benchmark on an EAF A100 slice (`core/eaf/eafhub.py run`) |

```bash
# GPVM: G4 reference (stock DS stage, ~12 ms/muon), then the fast arm on the same muons
bash beam/art/run_g4ref.sh $M2FS_DATA/beam_fastsim/g4ref_full 214310 1
bash beam/art/run_fast_cpu.sh $M2FS_DATA/beam_fastsim/g4ref_full $M2FS_DATA/beam_fastsim/fast_v1_20k 20000
# laptop/any: unit tests and benchmark
PYTHONPATH=core:beam python3 -m pytest -q beam/tests
PYTHONPATH=core:beam python3 -m beamfastsim.bench --n 2000
```

Status and validation: wiki `mu2e/wiki/sim/beam-muon-transport-fastsim.md`.
