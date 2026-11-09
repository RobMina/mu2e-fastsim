# mu2e-fastsim

**New here? [QUICKSTART.md](QUICKSTART.md)**: from this URL to your own job on EAF, a Mu2e GPVM or your own Slurm
cluster in about 10 minutes.

Geant4-free, GPU-ready fast simulation and fast reconstruction for Mu2e design and calibration studies:

- the **cosmic chain of the extracted (KPP, no field) configuration**: CRV (fast CrvSteps, response, digis, pulses,
  coincidences) and tracker (StrawGasSteps, digis, hits, seeds, line fit, EventNtuple), from the same CRY primaries;
- the **muon beam** from the detector-solenoid entrance to the stopping target (stop fraction and stop profile).

Each stage reproduces its stock Offline counterpart on identical inputs, so it can stand in for Geant4 + art in a
Mu2eBO optimisation loop or a MC/data study. One repository holds every stage, the project image and the shared docs.
Code only: no Mu2e-internal conditions, tables, data or results.

## Architecture

| Path | Package | What it holds |
|---|---|---|
| `core/` | `mu2efastsim` | physics primitives (Geant4-matched energy loss, materials, scattering, world, crossings), tables and statistics, CRY backends, endpoint adapters and the Mu2eBO kit `fastsimkit`, site configuration, test tooling |
| `crv/` | `crvfastsim` | CRV stage: transport, tier-1b secondary library, photons, SiPM charges, waveforms, pulses, coincidences |
| `tracker/` | `trkfastsim` | tracker stage: transport, secondaries, digitisation, hit reco, line finding and fit, EventNtuple |
| `calo/` | `calofastsim` | calorimeter response, propagation and hits (opt-in) |
| `beam/` | `beamfastsim` | beam stage: muon beam transport to the stopping target; the conversion-electron chain |
| `mixing/`, `trigger/` | | pileup mixing before digitisation (opt-in); the stock trigger-menu replay |
| `container/` | | the Apptainer image, the Offline-on-Slurm runner |

Layers: `core` imports no other package; the detectors (`crv`, `tracker`, `calo`) import `core`; the workflows
(`beam`, `mixing`, `trigger`) import anything (`tests/test_import_direction.py`). The core, crv and tracker histories
were merged in from their former repositories (`mu2e-fastsim-core`, `mu2e-crv-fastsim`, `mu2e-tracker-fastsim`);
`#NN` references in those commits point to pull requests there.

Every stage is NumPy by default with JAX twins for CPU and GPU; stages exchange flat tables keyed by
`(run, subrun, event)`. Diagram and ownership: [`docs/architecture.md`](docs/architecture.md).

## Status

| Stage | Validated against | Headline | Open |
|---|---|---|---|
| CRV | Geant4 + stock Offline, stage by stage and end to end | 5 M held-out events: counts within 0.25 %; cluster PE chi2/ndf 1.09 | cluster-PE shape, adjacent-layer correlations; named configurations |
| tracker | Geant4 + stock Offline | digis, hits and seeds agree with stock; track yield -1.42 +- 0.37 % | yield and rate offsets; secondary placement |
| beam | Geant4 DS stage on the same muons | stop fraction ratio 0.9976 +- 0.0058 (nominal), 0.9950 +- 0.0031 (49 foils) | 0.5-0.7 % low against refined-step Geant4 |
| electrons | ESTAR and Geant4 slabs; Geant4 CeEndpoint | CE yield at the tracker entry 1.004 +- 0.003 of Geant4; sob stand-in Geant4 / fast 0.9920 +- 0.0006 on 65 design points | energy-loss tail; calorimeter proxy |

Samples, thresholds, dates and commits: [`docs/validation.md`](docs/validation.md). Not implemented or not
validated: [`docs/limitations.md`](docs/limitations.md).

## Quickstart

[QUICKSTART.md](QUICKSTART.md) has the three paths (EAF, GPVM, own Slurm cluster), the demo
(`python3 -m mu2efastsim.demo`) and one kit task with a changed parameter. The inputs are Mu2e-internal: Mu2e members
request the input release `mu2e-fastsim-inputs-v1.2` as described in [`setup/README.md`](setup/README.md). The tests
need no inputs:

```bash
git clone https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
python3 -m venv .venv && .venv/bin/pip install -q -e core -e crv -e tracker -e beam -r core/ci/requirements.txt
.venv/bin/python -m pytest -q -p no:cacheprovider core/tests
```

Every environment option, `site.env` block and the beam staging step without CVMFS: [`docs/running.md`](docs/running.md).

## Endpoints and throughput

Throughput against stock Geant4 + Offline on a Fermilab grid node, as the number of single-core stock grid jobs one
device replaces at the same event rate (definitions, per-GPU tables and validity summaries:
[`docs/performance.md`](docs/performance.md)):

| Chain | One CPU core | One GPU (default precision) |
|---|---:|---:|
| CRV, cached CRY primaries to coincidence clusters | 40 | 280-810 (V100 to RTX PRO 6000; MIG slice 640-700) |
| tracker, cached CRY primaries to line fits | 149 | 700 (A100); 1,090 (MIG slice) |
| beam, DS entrance to target stops (49-foil design; stock starts at TS3) | not measured | 1,290 (MIG slice), 5,470 (A100) |

The Mu2eBO kit `fastsimkit` runs one point as one job on UVA (Slurm) or
EAF; tasks and setup in `core/docs/running.md`. On UVA a one-GPU job that asks for at most 32 GB runs on the free
RTX PRO 6000 MIG slice by default (`gpu_type="auto"`); larger requests keep their card.

**Device precision.** The CRV rows above run the device kernels in float32, tracker and beam in float64. The CRV device (JAX) kernels now run in float32 by
default (`MU2EFS_DEVICE_FLOAT`; float64 one variable away): within the float64 seed-to-seed spread on every validated
CRV quantity, mean pulse PE 0.03-0.14 % lower (bounded at 0.3 %, `crv/docs/device-precision.md`), 1.2-2.8x faster on
cards with reduced FP64 rate (MIG slices, A40, RTX PRO 6000). Device output repeats bit for bit from run to run on
one GPU kind at either precision because XLA's GPU autotuning is off by default (`MU2EFS_DEVICE_DETERMINISTIC`, about
+0.5 % warm time, faster compiles). Across GPU kinds float values differ in the last bits and float32 discrete counts
can differ slightly, so outputs record the precision, the determinism setting and the device model. Host NumPy code
and the beam and tracker JAX code stay float64.

## Configuration

`site.env.example` is the one site template of every package: every path, host and account comes from it;
nothing falls back to a built-in location, and a missing variable stops the command with its name. Each repository's
`tests/test_no_site_paths.py` (shared checker in core) keeps user names and site paths out of the code.

## Contributing and tests

Branches and pull requests only, with a green `tests` workflow (CI runs every suite with and without JAX) and a
review: [`CONTRIBUTING.md`](CONTRIBUTING.md). Cross-repository changes use one branch name in every repository.

## Studies kept off main

| Repository | Branch | Contents |
|---|---|---|
| umbrella | `park/beam-loss-studies` | beam loss diagnostics modules, tests and scripts |
| umbrella | `park/flash-celeritas` | Celeritas image, adapter and probes |
| core | `park/endpoint-esnet`, `park/beam-loss-studies` | ESnet endpoint; slab-split loss study |
| crv | `park/veto-closure`, `park/tier1b-library`, `park/response-diagnosis`, `park/named-run2`, `park/pvc-reco` | see the crv README |
| tracker | `park/f1-native-closure`, `park/ordered-navigation`, `park/rate-ledger`, `park/tv3-rescoring` | see the tracker README |

License: Apache-2.0 (`LICENSE`).
