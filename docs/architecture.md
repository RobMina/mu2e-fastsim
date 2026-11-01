# Architecture

## What is simulated

```
                    cosmic configuration (extracted / KPP, no field)                 beam
 CRY primaries ──┬──► CRV (crv/)                                       POT stage output (MuBeamCat muons)
 (gen tables,    │      transport -> tier-1b secondaries -> CrvSteps            │
  native or      │      -> photons -> SiPM charges -> waveforms -> digis        ▼
  device CRY)    │      -> pulses -> coincidence clusters               beam (beam/)
                 │                                                       DS field map (RK4), stopping-target
                 └──► tracker (tracker/)                                 foils, IPA: stops, decays, exits
                        world + interior transport -> StrawGasSteps            │
                        -> digis -> hits -> clusters, seeds -> line fit        ▼
                        -> EventNtuple                                   stop fraction and stop profile
```

Both cosmic stages read the same CRY primaries, so a tracker-CRV study uses one sample. Every stage reads and writes
flat tables keyed by `(run, subrun, event)` in the layout of the art dumpers, so any stage can be swapped for its stock
Offline counterpart (dumpers and table-to-art injectors in `crv/art/` and `tracker/art/`) and compared on the same
events. That is how everything is validated: one stage at a time against stock on identical inputs, then the chain
against Geant4 + stock.

## Repositories and ownership

| Repository | Path here | Owns |
|---|---|---|
| [mu2e-fastsim](https://github.com/RobMina/mu2e-fastsim) (this) | `.` | the pins, the beam stage (`beam/`), the project image (`container/`), the Offline-on-Slurm runner (`container/offline/`), the site template, the docs that span the stages |
| [mu2e-fastsim-core](https://github.com/RobMina/mu2e-fastsim-core) | `core/` | physics primitives (energy loss, materials, scattering, world, crossings), tables and statistics, CRY backends, endpoint adapters and the Mu2eBO kit, site configuration, test tooling and CI helpers |
| [mu2e-crv-fastsim](https://github.com/RobMina/mu2e-crv-fastsim) | `crv/` | the CRV stage (transport, tier 1b, response, reco), its art modules and validation |
| [mu2e-tracker-fastsim](https://github.com/RobMina/mu2e-tracker-fastsim) | `tracker/` | the tracker stage (transport, secondaries, digitisation, reco, EventNtuple), its art modules and validation |

Dependencies point one way: crv, tracker and beam import core (the tracker reads the CRV stage's gen tables, not its
code); core is meant to import none of the stages (the kit's CRV task code and one reverse import of `crvfastsim` are
still in core and are being moved out). Each repository has
`docs/architecture.md` for its own internals.

## Backends and devices

NumPy is the reference implementation of every stage and the default. JAX twins run the same algorithms on CPU,
NVIDIA GPUs (A100, B200, H100, V100, MIG slices) and Intel PVC (Aurora); they are validated statistically against NumPy
and are bit-identical per seed on one device. Device precision is float64.

## Compute endpoints

| Endpoint | Submission | Runtime |
|---|---|---|
| UVA Rivanna/Afton (Slurm) | ssh + Slurm (`core/endpoints/slurm_ssh.py`, `core/uva/`) | the Apptainer image (`container/`) |
| Fermilab EAF (A100 MIG slices) | JupyterHub API from a FNAL host (`core/endpoints/eaf.py`, `core/eaf/`) | a Python/JAX venv; Mu2e inputs read in place |
| ALCF Polaris (A100) | IRI API (`core/iri/`) | a venv on Eagle or the image |
| ALCF Aurora (Intel PVC) | ssh + PBS (`core/aurora/`) | a venv with `jax[oneapi]` |
| Mu2e GPVMs | interactive, one job per host | CVMFS `pyenv ana` (no JAX) |

The Mu2eBO kit `fastsimkit` (`core/mu2efastsim/endpoints/kit.py`) runs one Mu2eBO point as one job on UVA or EAF.
Tasks that read Mu2e-internal inputs stream them into node memory and delete them at exit (the kit refuses results
without the deletion record); tasks that read them in place run only on EAF.

## Configuration

Every site, user and storage location comes from the environment (`site.env.example`, one template for all four
repositories; read by `core/mu2efastsim/site.py`, which fails closed). Physics configuration is versioned in the
code: run configurations and tier-1b asset sets in crv, the response preset `crv/crvfastsim/configs/response_v1.json`,
the tracker digitisation conditions built from the stock job's sources.
