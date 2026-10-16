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

## Python environment

```bash
.venv/bin/python3 -m pip install -e core -e crv -e tracker
.venv/bin/python3 -m pytest core/tests crv/tests tracker/tests
# without installing (GPVM pyenv ana): source site.env, which sets PYTHONPATH=core:crv:tracker
```

## Site settings

Copy `site.env.example` to `site.env` (git-ignored) and source it before running job scripts.

License: Apache-2.0 (`LICENSE`).
