# Running

How to get from a fresh clone to one default shard of each chain. Every command reads its inputs from a verified
input release (`$M2FS_INPUTS`) and writes into an output area (`$M2FS_DATA`); nothing is written into the clone.

## 1. Clone

```bash
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git
cd mu2e-fastsim
git submodule update --init --recursive     # keep the recorded pins; never --remote in a reproduction
```

## 2. Python environment

Pick one:

| Where | How | JAX |
|---|---|---|
| any machine | `python3 -m venv .venv && .venv/bin/pip install -e core -e crv -e tracker -e beam -r core/ci/requirements.txt` | CPU JAX from the requirements; drop it for a NumPy-only setup |
| Mu2e GPVM | `source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && pyenv ana` | none (JAX tests skip; the beam stage needs JAX) |
| Slurm cluster | the project image (`container/mu2efastsim.def`, built with `container/build.sh`): `apptainer exec --cleanenv --env PYTHONPATH=... -B <code>,<inputs>,<outputs> mu2efastsim.sif python3 ...` | CPU or CUDA JAX |

The CI environment is `core/ci/requirements.txt` (Python 3.13, pinned numpy, pandas, uproot, awkward, scipy,
matplotlib, pyarrow, PyYAML, JAX). The image is not downloadable from these repositories yet.

## 3. Site configuration

```bash
cp site.env.example site.env        # or keep it outside the clone; it is git-ignored
# edit the placeholders of the blocks you use, then in every new shell:
source site.env
```

The minimum for the Python chains is the first block:

```bash
export M2FS_CODE="$PWD"                                  # this clone
export M2FS_INPUTS=/path/to/verified/input-release      # read-only
export M2FS_DATA=/path/to/writable/outputs              # on a Mu2e GPVM: your data area (quota), never the app area
export PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker:$M2FS_CODE/beam"
export CRVFS_DATA="$M2FS_INPUTS/crv_fastsim"
export CRVFS_LOOKUP_DIR="$CRVFS_DATA/lookup_v6_0"
export MU2E_SEARCH_PATH="$M2FS_INPUTS/offline"
```

A command that needs a variable you have not set stops with a message naming it (Python
`mu2efastsim.site.SiteConfigError`, shell `${VAR:?...}`); there are no per-user defaults. The other blocks:

| Block | Fill when you | Key variables |
|---|---|---|
| code and data | run anything | `M2FS_CODE` (this clone), `M2FS_INPUTS` (verified read-only release), `M2FS_DATA` (writable outputs), `PYTHONPATH` |
| hosts and accounts | use the kit or submit Slurm jobs | `FASTSIMKIT_FNAL_HOST`, `FASTSIMKIT_SLURM_HOST`, `FASTSIMKIT_SLURM_ACCOUNT`; `SBATCH_ACCOUNT` is read by `sbatch` for the repository's `.sbatch` files, which name no account |
| Slurm endpoint | run kit tasks on a Slurm cluster | `FASTSIMKIT_SCRATCH` (work area there), `FASTSIMKIT_SIF`, `FASTSIMKIT_WORKDIR`, the `FASTSIMKIT_*_CODE` snapshots, the `FASTSIMKIT_*_DATA` input roots on the FNAL host, `FASTSIMKIT_GEOMETRY_PREFIX` |
| Fermilab EAF | use the EAF endpoint | `FASTSIMKIT_EAF_SHARED_PREFIX`, `FASTSIMKIT_EAF_CODE`, `FASTSIMKIT_EAF_WORKDIR`, `FASTSIMKIT_EAF_VENV`, your own `EAF_TOKEN_FILE` |
| FNAL grid | submit grid jobs | `M2FS_PNFS` and the per-stage `*_PNFS`, `PNFS_OUT` for the beam closure, `CRVFS_OWNER`/`TRKFS_OWNER` |
| ALCF | run on Polaris or Aurora | `ALCF_PROJECT`, `M2FS_EAGLE`, `M2FS_FLARE`, `IRI`, `IRI_PY` |
| CRV, tracker, beam | run that stage's job scripts | `CRVFS_*`, `MU2E_SEARCH_PATH`, `TRKFS_*`, `BEAMFS_DATA_ROOT`, `VENV` |
| Offline runner | run art without host CVMFS (`container/offline/`) | `CVMFSEXEC_DIST`, `OFFLINE_ROOT`, `OFFLINE_AREA`, `OFFLINE_INPUT`, other `OFFLINE_*` |

Paths on the FNAL side (`M2FS_*`, `FASTSIMKIT_*_DATA`, `CRVFS_DATA`) are as seen on the FNAL host; Slurm-side paths
(`FASTSIMKIT_SCRATCH` and below) are as seen on the cluster. Keep secrets (tokens, keys) out of site.env; point to
your own mode-600 files instead. Do not commit a filled site.env.

## 4. Inputs

The chains read Mu2e-internal inputs (Geant4 reference tables, the tier-1b library, the GDML export, simulation
conditions, the CRV photon lookup tables, the beam muon table). They come as the input release
**`mu2e-fastsim-inputs-v1.2`**, readable by the Mu2e group on Fermilab storage from a GPVM or EAF. How to request
access, its layout and how to verify it against its checksum manifest are in [`setup/README.md`](../setup/README.md):

```bash
python3 -m mu2efastsim.inputs check-inputs --manifest RELEASE.json --root "$M2FS_INPUTS"
```

Exit 0 means every file is present with the recorded size and SHA-256.

## 5. Tests

```bash
python3 -m pytest -q -p no:cacheprovider core/tests
python3 -m pytest -q -p no:cacheprovider crv/tests          # run crv and tracker in separate processes
python3 -m pytest -q -p no:cacheprovider tracker/tests
python3 -m pytest -q -p no:cacheprovider beam/tests tests    # beam tests skip without JAX
```

No inputs are needed. `python3 -m mu2efastsim.testselect --also ../core --run --isolate` (from a stage directory)
runs only the test files a branch affects, one process per file. `--runslow` adds the large-sample validations.

## 6. One default shard per chain

### CRV (CPU, extracted_v04 default: `range_out` transport, tier 1b `strict_v5`)

```bash
G="$CRVFS_DATA/g4ref_v2_tables/gen.001400_00000022.root"; GEOM="$CRVFS_DATA/geometry"
python3 -m crvfastsim runchain "$G" --tier1b-dir "$CRVFS_DATA/tier1b_strict_v5" \
  --gdml "$GEOM/mu2e_extracted_v04.gdml" --calib "$GEOM/crvcalib_sim_best_v1_5_run1400.csv" \
  --deviation "$GEOM/crvphoton_sim_best_v1_5_run1400.csv" \
  --max-events 1000 --seed 22 --outdir "$M2FS_DATA/crv-smoke"
```

`$M2FS_DATA/crv-smoke/runchain_stats.json`: steps, digis, pulses and clusters per sector type and the cost per stage.
GPU options, the physics options and timing runs: `crv/docs/running.md`.

### Tracker (CPU, stage C primary-muon StrawGasSteps)

```bash
python3 -m trkfastsim steps "$G" --gdml "$GEOM/mu2e_extracted_v04.gdml" \
  --stage C --sliced-msc --seed 22 --outdir "$M2FS_DATA/tracker-smoke"
```

Digitisation, reco and the validation campaigns: `tracker/docs/running.md`.

### Beam (JAX on CPU or GPU, stop fraction of the nominal stopping target)

```bash
python3 -m beamfastsim.stage "$M2FS_DATA/beam_pub"       # public field map + geometry over HTTP; not needed where /cvmfs is mounted
export BEAMFS_OFFLINE_ROOT="$M2FS_DATA/beam_pub/offline" BEAMFS_MAPS="$M2FS_DATA/beam_pub/maps"   # as printed by the stage step
python3 -m beamfastsim point --g4 "$M2FS_INPUTS/beam_fastsim/g4ref_full/g4ref.npz" --n 20000 \
  --out "$M2FS_DATA/beam-smoke"
```

`$M2FS_DATA/beam-smoke/metrics.json`: stop fraction with its binomial error, stops per foil, stop radius.
`python3 -m beamfastsim.bench --n 2000` times the transport on synthetic muons with no inputs. Reference jobs and
comparisons: `beam/README.md`.

A smoke run tests the setup, not physics closure or throughput; do not extrapolate a runtime from it.

## 7. Endpoints and the Mu2eBO kit

`python3 -m mu2efastsim.endpoints.kit` is `fastsimkit`; `FASTSIMKIT_ENDPOINT=uva` (any Slurm site over ssh) or `eaf`.
Tasks and the `kits.toml` entry: `core/docs/running.md` and `core/uva/README.md`. The EAF CRV task is `chain`
(inputs read in place); `chain_point` (inputs streamed) runs on Slurm sites only. Art jobs (stock Offline) run on a
Slurm cluster without host CVMFS through `container/offline/` ([README](../container/offline/README.md)).
