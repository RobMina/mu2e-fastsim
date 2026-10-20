# mu2e-fastsim

Umbrella repository of the Mu2e Geant4-free, GPU-accelerated fast simulation and fast reconstruction for the
extracted (KPP, no-field) cosmic configuration. It pins compatible versions of three submodules:

| Submodule | Repository | What it holds |
|---|---|---|
| `core/` | [mu2e-fastsim-core](https://github.com/RobMina/mu2e-fastsim-core) | detector-independent physics and infrastructure: energy loss and fluctuations matched to Geant4, materials, GDML world, crossings, transport helpers, table I/O, statistics, ALCF IRI job drivers |
| `crv/` | [mu2e-crv-fastsim](https://github.com/RobMina/mu2e-crv-fastsim) | stage 1: CRV `CrvStep`s, GPU CRV response and fast CRV reconstruction |
| `tracker/` | [mu2e-tracker-fastsim](https://github.com/RobMina/mu2e-tracker-fastsim) | stage 2: tracker `StrawGasStep`s and fast tracker reconstruction |

Each stage is its own repository; this one records which commits work together and is the single working area: work in
the submodule checkouts here, commit and push in each submodule, then commit the new pins here. The CRV and tracker
packages import the shared code from `core/` (package `mu2efastsim`); there are no copies.

This repository holds technical software only: no physics results, no Mu2e-internal conditions, tables or data.

## Clone

```bash
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git
cd mu2e-fastsim
git submodule update --remote --merge     # move every submodule to its branch head, then commit the new pins
```

## Status (2026-10-20)

| stage | state | reproduce |
|---|---|---|
| CRV (`crv/`) | fast CrvStep transport, tier 1b secondaries, GPU CRV response and fast reco; validation in progress | `crv/README.md`, `crv/crvfastsim/README.md` |
| tracker (`tracker/`) | fast primary-muon StrawGasSteps: world transport, support-structure loss, in-tracker multiple scattering in support and straws with a single-scattering tail, G4-matched eIon, G4 secondary library; fast reco not started | `tracker/README.md` ("Reproducing the validation on a Mu2e GPVM") |

## Setup on a Mu2e GPVM

For Mu2e collaborators with an account on the Mu2e interactive machines. Nothing is built; the Python code runs in the
`ana` environment on CVMFS.

```bash
cd /exp/mu2e/app/users/$USER
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git
cd mu2e-fastsim
cp site.env.example site.env     # edit the first block: M2FS_CODE=$PWD, M2FS_DATA=/exp/mu2e/data/users/$USER
# every new shell:
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana
cd /exp/mu2e/app/users/$USER/mu2e-fastsim && source site.env      # PYTHONPATH=core:crv:tracker
# check (run the suites one at a time: crv/tests and tracker/tests share a test file name)
python3 -m pytest -q -p no:cacheprovider core/tests
python3 -m pytest -q -p no:cacheprovider tracker/tests
python3 -m pytest -q -p no:cacheprovider crv/tests      # a few JAX tests skip without JAX
```

Outputs go to `$M2FS_DATA` (your `/exp/mu2e/data/users/$USER`, 150 GiB quota), never into the clone. Run one job per
GPVM. The inputs (GDML export, G4 reference tables) are Mu2e-internal and not in the repositories; the stage READMEs
say where readable copies are and how they are produced.

## Python environment elsewhere

```bash
.venv/bin/python3 -m pip install -e core -e crv -e tracker
.venv/bin/python3 -m pytest core/tests; .venv/bin/python3 -m pytest crv/tests; .venv/bin/python3 -m pytest tracker/tests
```

## Site settings

Copy `site.env.example` to `site.env` (git-ignored) and source it before running anything. The first block (code and
data paths) is all the Python validation needs; the grid, ALCF and muse lines are for the job scripts of each stage.

License: Apache-2.0 (`LICENSE`).
