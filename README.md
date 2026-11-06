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
| CRV | Geant4 + stock Offline, stage by stage and end to end | V0, V1, V4, V-R1..V-R5 pass; 5 M held-out events: counts within 0.25 %; paired per-shard statistic: 11/155 histograms flagged with fast steps (7 quantities: cluster PE, multiplicities), 12/133 end to end | cluster-PE shape, adjacent-layer descendant correlation; Run 2 V1; named configurations |
| tracker | Geant4 + stock Offline | T-V0, T-V1 pass; digis, hits, seeds agree with stock | rate -0.82 +- 0.26 % (default), +0.14 +- 0.26 % with the opt-in `--scatter-in`; T-V3 yield -1.42 +- 0.37 % (default), -0.85 +- 0.37 % (`--scatter-in`); secondary placement; F1 frozen at 94.2 % (default), 95.6 % (opt-in time model) vs 99.5 % |
| beam | stock and refined-step Geant4 DS stage on the same muons | stop fraction ratio 0.9976 +- 0.0058 (nominal), 0.9950 +- 0.0031 (49 foils); finer Geant4 steps do not move it (refined / stock 1.0016 +- 0.0020, 1.0019 +- 0.0011) | fast 0.5-0.7 % low with paired errors, 2.15 sigma against refined steps on 49 foils; stopping-edge shape not compared for the default transport |
| electrons | ESTAR, Geant4 slabs (EM option 4); field stepping against analytic helices and RK4; conversion electrons to the tracker entry against Geant4 CeEndpoint | slab physics (`eslab`, device twin `eslab_jax` with field and lane compaction): loss chi2/ndf 0.6-1.4, 1 X0 W reflected 0.336 vs 0.347 +- 0.013 %; CE chain `beam/beamfastsim/cechain.py`: yield per generated electron 1.004 +- 0.003 and the angles and energy at the tracker entry agree with Geant4 CeEndpoint; sob stand-in `cesob`: Geant4 / fast 0.9920 +- 0.0006 on 65 design points; device chain (`--backend device`) agrees with it on the same starts | [core limitations](https://github.com/RobMina/mu2e-fastsim-core/blob/main/docs/limitations.md) |

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
python3 -m crvfastsim runchain "$G" --tier1b-dir "$CRVFS_DATA/tier1b_strict_v5" --gdml "$GEOM/mu2e_extracted_v04.gdml" \
  --calib "$GEOM/crvcalib_sim_best_v1_5_run1400.csv" --deviation "$GEOM/crvphoton_sim_best_v1_5_run1400.csv" \
  --max-events 1000 --seed 22 --outdir "$M2FS_DATA/crv-smoke"
# tracker, stage C steps of the same shard
python3 -m trkfastsim steps "$G" --gdml "$GEOM/mu2e_extracted_v04.gdml" --stage C --sliced-msc --seed 22 \
  --outdir "$M2FS_DATA/tracker-smoke"
# beam, stop fraction of the nominal target (JAX)
python3 -m beamfastsim point --g4 "$M2FS_INPUTS/beam_fastsim/g4ref_full/g4ref.npz" --n 20000 --out "$M2FS_DATA/beam-smoke"
```

Environment options (venv, Mu2e GPVM, image), every `site.env` block and the beam staging step without CVMFS:
[`docs/running.md`](docs/running.md). **The inputs are Mu2e-internal**: Mu2e members request the input
release `mu2e-fastsim-inputs-v1.1` as described in [`setup/README.md`](setup/README.md), which also gives its layout
and verification.

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
