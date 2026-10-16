# mu2e-crv-fastsim

Stage 1 of the umbrella repository [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim) (submodule `crv/`). The
detector-independent code (energy loss, materials, world, generic crossings/transport/tables, statistics, plots, key
matching and the ALCF IRI job tools) lives in [`mu2e-fastsim-core`](https://github.com/RobMina/mu2e-fastsim-core)
(package `mu2efastsim`); this repository holds only the CRV-specific code and depends on it.

A Geant4-free, GPU-accelerated simulation of the Mu2e KPP (extracted) Cosmic Ray Veto cosmic sample, for MC/data
agreement studies. It runs in a container on ALCF Polaris and is submitted through the ALCF IRI API. The CRV comes
first; the tracker follows.

| path | content |
|---|---|
| `docs/plan.md` | scope, measured baseline, agreed performance goals and validation thresholds, architecture, schedule |
| `docs/schema_and_transport.md` | CrvStep-table design notes and the muon-transport design |
| `crvfastsim/` | the Python package and CLI (`python3 -m crvfastsim`); **reference: `crvfastsim/README.md`** |
| `tests/` | pytest unit tests (from the umbrella root: `PYTHONPATH=core:crv python3 -m pytest -q crv/tests`) |
| `art/` | muse area with the `CrvFastSim` art package (`CrvStepTableDumper`) and `dump_g4ref.sh` |
| `grid/` | FermiGrid jobs for the G4 reference sample (`g4ref_v1`) |
| `geometry/` | Offline job that exports the G4 world (GDML + bar printout) for `crvfastsim v0check` |
| `profile/` | Offline job configs and driver of the 2026-10-15 baseline profile; summaries in `profile/results/` |
| `containers/` | Polaris image recipe (not started) |
| `site.env.example` | site paths and accounts for the job scripts; copy to `site.env` (git-ignored) and source it |

Python conventions follow `crvextrap` (`RobMina/mu2e_track_crv_mcresponse`): one package, every analysis a subcommand,
shared definitions in one module, vectorised hot paths, `--emit`/`--merge` batching, stats JSON beside the figures,
no loose one-off scripts.

## License and data

Apache-2.0 (`LICENSE`, `NOTICE`), as Mu2e Offline. The repository holds code, configuration and the agreed plan only:
Mu2e-internal inputs (the CRV conditions and photon lookup tables, detector data and anything derived from it) and
simulation outputs are never committed. Physics results from this work are reported through Mu2e documents.
