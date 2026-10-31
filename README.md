# mu2e-fastsim

## Collaborator first run

Start with [setup/README.md](setup/README.md): explicit input-release and output roots,
checksum verification, one-shard CRV/tracker recipes and endpoint prerequisites.
There is no approved shared input release bundled here; another user's data area is not a collaborator data
source. Keep the recorded submodule pins when reproducing a run. Every site, user and storage path comes from your
`site.env` (see `site.env.example`); nothing in the code falls back to a built-in location.

Umbrella repository of the Mu2e Geant4-free, GPU-accelerated fast simulation and fast reconstruction: the extracted
(KPP, no-field) cosmic configuration (CRV and tracker stages), and the muon beam from the detector-solenoid entrance to
the stopping target (`beam/`). It pins compatible versions of three submodules and holds the beam stage itself:

| Submodule | Repository | What it holds |
|---|---|---|
| `core/` | [mu2e-fastsim-core](https://github.com/RobMina/mu2e-fastsim-core) | detector-independent physics and infrastructure: energy loss and fluctuations matched to Geant4, materials, GDML world, crossings, transport helpers, table I/O, statistics, compute-endpoint drivers and the Mu2eBO kit |
| `crv/` | [mu2e-crv-fastsim](https://github.com/RobMina/mu2e-crv-fastsim) | stage 1: CRV `CrvStep`s, GPU CRV response and fast CRV reconstruction |
| `tracker/` | [mu2e-tracker-fastsim](https://github.com/RobMina/mu2e-tracker-fastsim) | stage 2: tracker `StrawGasStep`s and fast tracker reconstruction |

| Directory (in this repository) | What it holds |
|---|---|
| `beam/` | package `beamfastsim`: muon transport from the DS entrance through the field map and the stopping-target foils (stops, decays, exits), the Geant4 reference job and the fast-vs-Geant4 comparison; see `beam/README.md` |

Each stage is its own repository; this one records which commits work together and is the single working area: work in
the submodule checkouts here, commit and push in each submodule, then commit the new pins here. The CRV and tracker
packages import the shared code from `core/` (package `mu2efastsim`); there are no copies.

This repository holds technical software only: no physics results, no Mu2e-internal conditions, tables or data.

## Clone

```bash
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git
cd mu2e-fastsim
git submodule update --init --recursive   # keep the compatible recorded pins
```

## Status (2026-10-23)

| stage | state | reproduce |
|---|---|---|
| CRV (`crv/`) | fast CrvStep transport, tier 1b secondaries, fused GPU CRV response (photons -> SiPM charges) and fast reco (waveforms, pulses, coincidences) in numpy and JAX. The fast chain (fast response + fast reco) reproduces the stock chain from the same G4 steps (V-R4), and the CRV response objective has the same minimum from both paths on a knob scan (V-R5). One objective point on 1 M cached events runs end to end in 10.4 min on one Polaris node (R3). An out-of-time background model resampled from KPP data (`crvfastsim.reco.background`) is the newest piece | `crv/README.md`, `crv/crvfastsim/README.md`, `crv/docs/fastreco.md` |
| tracker (`tracker/`) | fast primary-muon StrawGasSteps: world transport, support-structure loss, in-tracker multiple scattering in support and straws with a single-scattering tail, G4-matched eIon, G4 secondary library. Fast reco started (`trkfastsim.reco`, numpy with JAX twins): straw-hit and combo-hit reco reproduce stock on the same digis (T-VR1), time clusters and line seeds too (T-VR1b); digitisation and the track fit are next | `tracker/README.md` ("Reproducing the validation on a Mu2e GPVM") |
| beam (`beam/`) | muon transport from the DS entrance (stock `MuBeamCat` muons) through the DS field map (RK4, trilinear lookup as Offline), the stopping-target foils and the IPA, with Geant4-matched energy loss (`mu2efastsim.eloss_jax`), Highland scattering, range-out and decay in flight, in JAX on CPU or GPU. Stopping fraction and stop profile agree with the stock Geant4 DS stage run on the same muons, for the nominal target and a 49-foil geometry move; one A100 MIG slice replaces the order of a hundred grid slots for this stage | `beam/README.md` |

## Compute endpoints and the Mu2eBO kit

| endpoint | submission | runtime | state | drivers |
|---|---|---|---|---|
| ALCF Polaris (A100) | IRI API | a venv on Eagle (`M2FS_EAGLE`) or the image | reference for the performance goals | `core/iri/` |
| ALCF Aurora (Intel PVC) | SSH + PBS | venv with `jax[oneapi]` | CRV response runs; fast reco does not port yet | `core/aurora/` |
| ESnet East (AWS) | IRI API v2 | OCI image as Kubernetes pods | CPU and T4 pods run | `core/iri/` (`IRI_FACILITY=esnet`) |
| UVA Rivanna/Afton (A100, B200) | SSH + Slurm | the Apptainer image | in-image validation passes on A100 and B200; production as one-GPU jobs | `core/uva/` |
| Fermilab EAF (A100 MIG slices) | JupyterHub API, driven from a GPVM | a JAX venv (`FASTSIMKIT_EAF_VENV`) | Mu2eBO endpoint: queue, auto-stop, keepalive; Mu2e inputs stay at FNAL | `core/eaf/` |

`fastsimkit` (`core/mu2efastsim/endpoints/kit.py`) is the [Mu2eBO](https://github.com/Mu2e/Mu2eBO) native kit for
the fast chain: it speaks the evaluator contract (`submit`/`status`/`results`/`check`/`describe`/`cancel`) and runs
one point as one job through an endpoint adapter (`core/mu2efastsim/endpoints/`: Slurm over SSH, and Fermilab EAF).
Two Mu2eBO entries use it: `fastsimkit` (UVA) and `fastsimkit_eaf` (EAF, one A100 MIG slice with a queue).

| Task | What it returns | Endpoints |
|---|---|---|
| `reco_bench` | synthetic CRV reco chain throughput | any GPU |
| `response_bench` | photon and SiPM throughput with the real lookup table and photon map | any GPU |
| `v4_response` | the V4 photon and SiPM checks (pass flags, largest deviation) | any GPU |
| `chain_point` | one point of the whole CRV fast chain as N one-GPU jobs with the inputs streamed in | UVA |
| `chain`, `regression` | the CRV fast chain on held-out G4 shards vs the stock chain; the regression set | EAF |
| `veto_c4` | CRV veto inefficiency of a veto card on a cached Run 1A selection (CPU) | EAF |
| `beam_point` | stopping fraction and stop profile of a stopping-target geometry (`beam/`) | EAF |

Tasks that use the Mu2e-internal tables fetch them per job into node memory and delete them at exit, and the kit
refuses results without the job's deletion record; tasks that read Mu2e-internal inputs in place run only on EAF, so
nothing leaves Fermilab. `beam_point` is registered when `beam/` is on `PYTHONPATH`. The `kits.toml` entries are in
`core/uva/README.md` and in the Mu2eBO repository.

## Setup on a Mu2e GPVM

For Mu2e collaborators with an account on the Mu2e interactive machines. Nothing is built; the Python code runs in the
`ana` environment on CVMFS.

```bash
cd "$YOUR_APP_AREA"              # your Mu2e app (code) area on the interactive machines
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git
cd mu2e-fastsim
cp site.env.example site.env     # edit M2FS_CODE, approved M2FS_INPUTS, writable M2FS_DATA and site paths
# every new shell:
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana
cd "$YOUR_APP_AREA/mu2e-fastsim" && source site.env      # PYTHONPATH=core:crv:tracker:beam
# check (run the suites one at a time: crv/tests and tracker/tests share a test file name)
python3 -m pytest -q -p no:cacheprovider core/tests
python3 -m pytest -q -p no:cacheprovider tracker/tests
python3 -m pytest -q -p no:cacheprovider crv/tests      # a few JAX tests skip without JAX
python3 -m pytest -q -p no:cacheprovider beam/tests     # needs JAX (the beam stage has no numpy fallback)
```

**Which tests to run.** A change rarely needs every suite. From a stage's directory, `python3 -m mu2efastsim.testselect`
lists the test files that import, directly or transitively, a module changed since the merge base with `origin/main`
(`--also ../core` counts core changes too; `--why` gives the reason per file). `--run --isolate` runs them one pytest
process per file, so peak memory is that of the heaviest file (at most about 1.7 GB in core, crv and tracker on a
laptop) instead of the whole suite (about 3 GB for `crv/tests` in one process). Large-sample statistical validations are marked `@pytest.mark.slow` and skipped
unless `--runslow` is given; run them when the code they validate changes (`--run --isolate -- --runslow`).

```bash
cd crv && python3 -m mu2efastsim.testselect --also ../core --why                  # what a branch affects
python3 -m mu2efastsim.testselect --also ../core --run --isolate -- -q            # run it, one file per process
python3 -m pytest -q -p no:cacheprovider tests --runslow                          # everything, slow checks included
```

Outputs go to `$M2FS_DATA` (your Mu2e data area, which has a quota), never into the clone. Run one job per
GPVM. The inputs (GDML export, G4 reference tables) are Mu2e-internal and not in the repositories; the stage READMEs
describe their production. Readable collaborator publication is a separate gate; see setup/README.md.

## Python environment elsewhere

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -e core -e crv -e tracker
.venv/bin/python3 -m pytest core/tests; .venv/bin/python3 -m pytest crv/tests; .venv/bin/python3 -m pytest tracker/tests
PYTHONPATH=core:beam .venv/bin/python3 -m pytest beam/tests                 # beam/ is not an installable package yet
```

## Site settings

Copy `site.env.example` to `site.env` (git-ignored, or keep it outside the clone) and source it before running
anything; `setup/README.md` explains each block. The first block (code and data paths) is all the Python validation
needs; the endpoint, grid, ALCF, Offline-runner and muse blocks are for the job scripts of each stage. Code that needs
a variable you have not set stops with a message naming it.

License: Apache-2.0 (`LICENSE`).
