# mu2e-crv-fastsim

Stage 1 of the umbrella repository [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim) (submodule `crv/`). The
detector-independent code (energy loss, materials, world, generic crossings/transport/tables, statistics, plots, key
matching and the ALCF IRI job tools) lives in [`mu2e-fastsim-core`](https://github.com/RobMina/mu2e-fastsim-core)
(package `mu2efastsim`); this repository holds only the CRV-specific code and depends on it.

A Geant4-free, GPU-accelerated simulation and fast reconstruction of the Mu2e KPP (extracted) Cosmic Ray Veto cosmic
sample, for MC/data agreement studies: fast CrvSteps, the fused GPU CRV response, waveforms, pulse and coincidence
reco, and the CRV response objective, in numpy and JAX. It runs on ALCF Polaris (IRI API) and Aurora, ESnet, UVA and
Fermilab EAF through the endpoint drivers in `mu2e-fastsim-core`, and as the Mu2eBO kit `fastsimkit`. Validation
status: the umbrella README; fast reco goals and results: `docs/fastreco.md`.

| path | content |
|---|---|
| `docs/plan.md` | scope, measured baseline, agreed performance goals and validation thresholds, architecture, schedule |
| `docs/fastreco.md` | fast reconstruction: goals R1-R6, thresholds V-R1..V-R5, results (V-R4, V-R5, R3, the out-of-time background model) |
| `docs/agent13-workflow.md` | experimental staged full-chain benchmarking, exact output contracts and opt-in dispatch optimization |
| `docs/response-configuration.md` | schema-v1 JSON response/reco configuration, units/support policy, CLI/kit overrides and resolved asset provenance |
| `docs/tier1b-default.md` | the tier-1b default asset set (`strict_v3`), `strict_v2` and the `legacy` option: files, options, named-configuration caveat, equivalence |
| `docs/step-layer-closure.md` | opt-in tier-1b placement, held-out shower-source diagnostic, remaining closure and identity limits |
| `docs/tier1b-descendant-audit.md` | descendant expectation/migration, factorial DQM controls, independent shower closure and candidate-bank support |
| `docs/generator-association-repair.md` | strict donor association schema, completed production repair and held-out physics validation |
| `docs/corrected-shower-uncertainty.md` | exact shower sampler moments, donor-file uncertainty, corrected source contrast and exposure census |
| `docs/schema_and_transport.md` | CrvStep-table design notes and the muon-transport design |
| `crvfastsim/` | the Python package and CLI (`python3 -m crvfastsim`); **reference: `crvfastsim/README.md`** |
| `tests/` | pytest unit tests (from the umbrella root: `PYTHONPATH=core:crv python3 -m pytest -q crv/tests`) |
| `art/` | muse area with the `CrvFastSim` art package (`CrvStepTableDumper`) and `dump_g4ref.sh` |
| `grid/` | FermiGrid jobs: the G4 reference sample (`g4ref_v1`), V-R4 and V-R5 stock arms, CRY generator tables (`grid/README.md`) |
| `geometry/` | Offline job that exports the G4 world (GDML + bar printout) for `crvfastsim v0check` |
| `profile/` | Offline job configs and driver of the 2026-10-15 baseline profile; summaries in `profile/results/` |
| `scripts/` | GPVM drivers (reco checks, V-R4 runs, the background-model fit) |
| `site.env.example` | site paths and accounts for the job scripts; copy to `site.env` (git-ignored) and source it |

Python conventions follow `crvextrap` (`RobMina/mu2e_track_crv_mcresponse`): one package, every analysis a subcommand,
shared definitions in one module, vectorised hot paths, `--emit`/`--merge` batching, stats JSON beside the figures,
no loose one-off scripts.

## Native CRY (opt-in)

`runchain native --gen native --gen-events N` can generate extracted primaries
in-process with the CRY-enabled container and current core wrapper. It removes
the gen-table dependency only; other chain inputs remain required. See
[native CRY](docs/native-cry.md) for setup, emitted/merged exposure receipts,
core's extracted-only `chain_point gen_set=4`, and validation limits.
`--gen device` / core `chain_point gen_set=5` select JAX on-device generation.
Table-driven defaults are unchanged.

## License and data

Apache-2.0 (`LICENSE`, `NOTICE`), as Mu2e Offline. The repository holds code, configuration and the agreed plan only:
Mu2e-internal inputs (the CRV conditions and photon lookup tables, detector data and anything derived from it) and
simulation outputs are never committed. Physics results from this work are reported through Mu2e documents.
