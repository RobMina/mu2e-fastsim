# mu2e-fastsim

Geant4-free, GPU-ready fast simulation and fast reconstruction for Mu2e design and calibration studies:

- the **cosmic chain of the extracted (KPP, no field) configuration**: CRV (fast CrvSteps, response, digis, pulses,
  coincidences) and tracker (StrawGasSteps, digis, hits, seeds, line fit, EventNtuple), from the same CRY primaries;
- the **muon beam** from the detector-solenoid entrance to the stopping target (stop fraction and stop profile).

Each stage reproduces its stock Offline counterpart on identical inputs, so it can stand in for Geant4 + art in a
Mu2eBO optimisation loop or a MC/data study. This umbrella pins compatible commits of three submodules and holds the
beam stage, the project image and the shared docs. Code only: no Mu2e-internal conditions, tables, data or results.

## Architecture

| Repository | Path | What it holds |
|---|---|---|
| [mu2e-fastsim-core](https://github.com/RobMina/mu2e-fastsim-core) | `core/` | physics primitives (Geant4-matched energy loss, materials, scattering, world, crossings), tables and statistics, CRY backends, endpoint adapters and the Mu2eBO kit `fastsimkit`, site configuration, test tooling |
| [mu2e-crv-fastsim](https://github.com/RobMina/mu2e-crv-fastsim) | `crv/` | CRV stage: transport, tier-1b secondary library, photons, SiPM charges, waveforms, pulses, coincidences |
| [mu2e-tracker-fastsim](https://github.com/RobMina/mu2e-tracker-fastsim) | `tracker/` | tracker stage: transport, secondaries, digitisation, hit reco, line finding and fit, EventNtuple |
| this repository | `beam/`, `container/` | beam stage (`beamfastsim`), the Apptainer image, the Offline-on-Slurm runner |

Every stage is NumPy by default with JAX twins for CPU and GPU; stages exchange flat tables keyed by
`(run, subrun, event)`. Diagram and ownership: [`docs/architecture.md`](docs/architecture.md).

## Status

| Stage | Validated against | Result | Open |
|---|---|---|---|
| CRV | Geant4 + stock Offline, stage by stage and end to end | V0, V1, V4, V-R1..V-R5 pass; 5 M held-out events: counts within 0.25 %, 21/155 histograms flagged (steps), 13/133 end to end | cluster-PE shape, adjacent-layer descendant correlation; Run 2 V1; named configurations |
| tracker | Geant4 + stock Offline | T-V0, T-V1 pass; digis, hits, seeds agree with stock | rate -0.89 +- 0.32 %; T-V3 yield -2.04 +- 0.37 %; F1 95.6 % vs 99.5 % |
| beam | stock Geant4 DS stage on the same muons | stop fraction ratio 0.9976 +- 0.0058 (nominal), 0.9950 +- 0.0031 (49 foils) | loss model at the stopping edge |
| electrons | ESTAR, Geant4 thin layers | on a core branch, not on `main` | multiple scattering, annihilation, field |

Tables with samples, thresholds, dates and commits: [`docs/validation.md`](docs/validation.md). Not implemented or
not validated: [`docs/limitations.md`](docs/limitations.md).

## Quickstart

```bash
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
cp site.env.example site.env      # set M2FS_INPUTS (verified input release) and M2FS_DATA (outputs)
source site.env
python3 -m pytest -q -p no:cacheprovider core/tests          # no inputs needed
python3 -m mu2efastsim.inputs check-inputs --manifest RELEASE.json --root "$M2FS_INPUTS"
G="$CRVFS_DATA/g4ref_v2_tables/gen.001400_00000022.root"; GEOM="$CRVFS_DATA/geometry"
# CRV, default chain on 1,000 events (CPU)
python3 -m crvfastsim runchain "$G" --tier1b-dir "$CRVFS_DATA/tier1b_strict_v2" --gdml "$GEOM/mu2e_extracted_v04.gdml" \
  --calib "$GEOM/crvcalib_sim_best_v1_5_run1400.csv" --deviation "$GEOM/crvphoton_sim_best_v1_5_run1400.csv" \
  --max-events 1000 --seed 22 --outdir "$M2FS_DATA/crv-smoke"
# tracker, stage C steps of the same shard
python3 -m trkfastsim steps "$G" --gdml "$GEOM/mu2e_extracted_v04.gdml" --stage C --sliced-msc --seed 22 \
  --outdir "$M2FS_DATA/tracker-smoke"
# beam, stop fraction of the nominal target (JAX)
python3 -m beamfastsim point --g4 "$M2FS_INPUTS/beam_fastsim/g4ref_full/g4ref.npz" --n 20000 --out "$M2FS_DATA/beam-smoke"
```

Environment options (venv, Mu2e GPVM, image), every `site.env` block and the beam staging step without CVMFS:
[`docs/running.md`](docs/running.md). **The inputs are Mu2e-internal and not yet published as a collaborator
release**; [`setup/README.md`](setup/README.md) gives the release layout and its verification.

## Endpoints and throughput

CRV default chain, cached CRY primaries to coincidence clusters, warm, float64, events with CRV steps per second:

| Endpoint | Unit | Events/s |
|---|---|---:|
| stock Offline | one grid job (one core) | 0.92 |
| fast, CPU | one core | 55 |
| Fermilab EAF | one A100 MIG 1g.10gb slice | 163 |
| ALCF Polaris | one node, 4 A100 | 990 |
| ALCF Aurora | one node, 12 PVC tiles | 1,212 |
| UVA | 8 A100 (one per job) | 1,695 |

Beam: 14,260 muons/s on a MIG slice, 42,270 on an A100 (Geant4: 83 per core). Tracker F2 fit: 57,000 tracks/s on a
MIG slice (stock KKLine about 26 per core). The Mu2eBO kit `fastsimkit` runs one point as one job on UVA (Slurm) or
EAF; tasks and setup in `core/docs/running.md`.

## Configuration

`site.env.example` is the one site template of all four repositories: every path, host and account comes from it;
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
