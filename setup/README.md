# Collaborator inputs and first run

## Current availability gate

A fresh clone is not a data release. No collaboration-readable input release or
redistribution approval is bundled with this code. Obtain an **approved versioned
manifest and source** from the input publisher before running the real chain.
Do not link another user's private data area, assume its ACLs, or use a scratch
cache as the canonical release. The Apache-2.0 code licence does not grant rights
to conditions, simulation-derived banks, detector data, CRY data, or an image's
third-party dependencies.

Proposed release layout (paths relative to `$M2FS_INPUTS`):

| Path | Needed by | Reproduction/provenance |
|---|---|---|
| `crv_fastsim/tier1b_strict_v2/{library.root,topology_sidecar_strict_v2_v2.npz,topology_sidecar_strict_v2_v2.npz.json,templates_v3.npz,templates_v3.npz.json}` | default CRV tier1b-apply, runchain, e2e-bench, kit | CRV docs/generator-association-repair.md, descendant-placement-closure.md, descendant-samebar-closure.md, tier1b-default.md; donor G4 production and versioned builders |
| `crv_fastsim/geometry/mu2e_extracted_v04.gdml` | CRV and tracker transport | CRV geometry/ Offline export; pin release and configuration |
| `crv_fastsim/geometry/{crvcalib_sim_best_v1_5_run1400.csv,crvphoton_sim_best_v1_5_run1400.csv}` | CRV response/reco | simulation run-1400 conditions; CRV docs/response-configuration.md; export needs conditions access |
| `crv_fastsim/lookup_v6_0/{LookupTable_6000_0,LookupTable_2370_1,LookupTable_3200_0,photonMap.root}` | CRV response (not tier1b-apply alone) | CVMFS DataFiles/CRVConditions/v6_0; names from extracted response configuration; Mu2e-internal, publisher approval needed |
| `offline/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt` and `offline/Offline/CRVResponse/data/singlePEWaveform_v3.txt` | CRV geometry and waveform | public Mu2e/Offline tag v13_39_00; container/fetch_offline_data.sh; included in standard image |
| `crv_fastsim/g4ref_v2_tables/gen.001400_00000022.root` | one-shard CRV/tracker smoke | extracted CRY reference shard 22, 50000 generated events; crv/grid and art/dump_g4ref.sh; use only a manifest-selected shard |
| `trk_fastsim/lib/trklib_mdc2025as_20f_v2.root` (+ metadata) | optional tracker secondaries | tracker buildlib format 2; G4 production donors; not read by primary-only steps |
| `trkfs_digi_20261026/v2/digicond.npz` | tracker_point digi | tracker digi-cond from FHiCL/conditions/GDML; geometry/release must match |
| `trk_fastreco/cond_mc_au.json` and `trk_fastreco/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root` (conditions tree only) | tracker_point reco/drift | tracker reco-conditions from matched dumper; kit_task.INPUT_PATHS is authoritative; no stock event conditioning |
| `beam_fastsim/g4ref_full/g4ref.npz` | beam point | MuBeamCat input and beam/art/run_g4ref.sh, beamfastsim g4ref and reference audit; no public redistributable source established |
| beam `offline/` geometry include tree, `maps/DSMap.{header,bin}` | beam point | beamfastsim.stage: Offline v10_47_00 and DataFiles/BFieldMaps/Mau13; pin every fetched file |

Native CRY is **opt-in**, not the default. It removes gen tables only, not the
secondary bank, geometry, optics or conditions. It requires the CRY-enabled
runtime and licensed data; see core/docs/native-cry.md. G4 comparisons still need
matching reference tables. The named non-extracted configurations currently need
the unavailable legacy library; do not silently substitute the extracted bank.

## Fresh clone and explicit site configuration

```bash
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git
cd mu2e-fastsim
# Keep the recorded submodule pins; do not update --remote in a reproduction recipe.
cp site.env.example site.env       # or keep it outside the clone
# Replace every placeholder in the blocks you use; then, in every new shell:
source site.env
```

### What to fill in site.env

`site.env.example` lists every variable the umbrella, `core/`, `crv/`, `tracker/` and `beam/` read, with
placeholder values only. The repositories contain no user names, home or storage paths, Slurm accounts or host
names: they come from this file (or from command-line arguments). A command that needs an unset variable stops
with a message naming it (Python `mu2efastsim.site.SiteConfigError`, shell `${VAR:?...}`); there are no per-user
defaults. Public locations (CVMFS, relative paths, `~/.mu2efastsim` tool state) remain defaults.

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
| Celeritas probes | rerun the GPU probes (`container/celeritas_adapter/`) | `CELERITAS_SIF`, `CELERITAS_PROBE_OUT`, `CELERITAS_BUILD_ROOT`, the per-probe roots and builds |

Paths on the FNAL side (`M2FS_*`, `FASTSIMKIT_*_DATA`, `CRVFS_DATA`) are as seen on the FNAL host; Slurm-side paths
(`FASTSIMKIT_SCRATCH` and below) are as seen on the cluster. Keep secrets (tokens, keys) out of site.env; point to
your own mode-600 files instead. Do not commit a filled site.env.

On a Mu2e GPVM use `source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh`
then `pyenv ana` before sourcing `site.env`. On UVA use an approved, checksum-pinned
Apptainer image and a scheduler allocation, not the login node. The image must be
readable to the collaborator; it is not downloadable from this repository.
Use `apptainer exec --cleanenv --env PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker:$M2FS_CODE/beam"`
with explicit bindings for code, inputs and outputs. Pass the site variables into
the clean environment explicitly, or source the edited site.env inside it.
A different HOME with the same uid tests hidden configuration, **not ACLs**.

## Verify before use; fetch is explicit and dry-run by default

The manifest is JSON schema version 1. Every file records `path` (safe relative
path), `size_bytes`, `sha256`, `provenance`, `redistribution` (`approved` only
after review). It must be supplied through a trusted publisher; a checksum does
not authenticate a malicious manifest. `inventory` hashes explicit sources and
records metadata, without copying inputs. Do not commit internal manifests or
source paths to the public repository.

```bash
python3 -m mu2efastsim.inputs check-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS"
python3 -m mu2efastsim.inputs fetch-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS" \
  --source-dir "$APPROVED_SOURCE"                     # dry-run; exit 1 if anything is missing
# Only after source, location and redistribution approval:
python3 -m mu2efastsim.inputs fetch-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS" \
  --source-dir "$APPROVED_SOURCE" --execute
```

Alternatively use `--base-url https://publisher.example/release` for approved
HTTPS objects. This tool does not mint credentials or implement Rucio/dCache
authentication: use the experiment's approved client to populate a source release,
then verify it. Fetch refuses mismatched existing files, traversal and symlink
escapes; verifies temporary bytes and publishes without overwriting. Do not use
it in a directory concurrently writable by untrusted users. Exit codes: 0 all
verified/fetched, 1 missing/mismatched or dry-run work remains, 2 invalid manifest,
I/O error or refused fetch. Missing hashes are never treated as verification.

## CRV: one manifest-selected shard, CPU reference

```bash
G="$CRVFS_DATA/g4ref_v2_tables/gen.001400_00000022.root"
GEOM="$CRVFS_DATA/geometry"
mkdir -p "$M2FS_DATA/crv-smoke"
python3 -m crvfastsim runchain "$G" --tier1b strict_v4 \
  --tier1b-dir "$CRVFS_DATA/tier1b_strict_v2" --gdml "$GEOM/mu2e_extracted_v04.gdml" \
  --calib "$GEOM/crvcalib_sim_best_v1_5_run1400.csv" \
  --deviation "$GEOM/crvphoton_sim_best_v1_5_run1400.csv" \
  --max-events 1000 --seed 22 --outdir "$M2FS_DATA/crv-smoke/chain"
python3 -m crvfastsim e2e-bench "$G" --tier1b strict_v4 \
  --tier1b-dir "$CRVFS_DATA/tier1b_strict_v2" --gdml "$GEOM/mu2e_extracted_v04.gdml" \
  --calib "$GEOM/crvcalib_sim_best_v1_5_run1400.csv" \
  --deviation "$GEOM/crvphoton_sim_best_v1_5_run1400.csv" \
  --max-events 1000 --seed 22 --repeat 2 --outdir "$M2FS_DATA/crv-smoke/bench"
```

Remove `--max-events` for the full shard. A smoke run tests setup, not physics
closure or throughput. Do not infer runtime from this event limit. Set
`CRVFS_LOOKUP_DIR` and `MU2E_SEARCH_PATH` as in site.env. With a verified local
Offline tree, `CRVFS_NO_FETCH=1` makes the geometry helper refuse hidden fetching.

## Tracker: one shard, primary steps

```bash
python3 -m trkfastsim steps "$G" --stage C --sliced-msc --seed 22 \
  --gdml "$GEOM/mu2e_extracted_v04.gdml" --outdir "$M2FS_DATA/tracker-smoke"
# To include secondaries, add only a matching verified library:
# --library "$M2FS_INPUTS/trk_fastsim/lib/trklib_mdc2025as_20f_v2.root"
```

This is the primary transport chain, not full tracker digitisation/track reco.
The latter needs the separately matched digi conditions and reco inputs in
tracker/docs/fast-digi.md and the tracker README. No G4/art build is required for
the Python steps command.

## EAF and the kit: distinguish the tasks

`chain_point` is the **streamed Slurm/UVA** task and is not implemented on EAF.
On EAF use task `chain` (in-place input reads), not `chain_point`. No successful
EAF deployment or clean-uid data access is implied by these instructions.
From a FNAL-network host, provide your own mode-600 `EAF_TOKEN_FILE`, readable
clone, verified input release and EAF Python/JAX venv; set the `FASTSIMKIT_EAF_*`
variables from site.env, then run the kit with `FASTSIMKIT_ENDPOINT=eaf`.
The in-place task `chain` currently reads g4ref_v1 shard 11 for `shards=1`;
that gen table (and optional matched `v2_fastC_v1/ref...11.root`) is an additional
manifest input, not the v2 shard 22 used in the CPU recipe.

Example evaluator parameters: `{"task":"chain","shards":1,"max_events":1000,"seed":22}`.
Use the kit's `check` before `submit`, then `status`/`results`; do not confuse
admission/render tests with a live job. The image is for Slurm jobs, while EAF
uses its configured venv. Endpoint access, allocation and token authorisation
remain site-specific.

## Validation drivers

Some stage validation drivers write under `$CRVFS_DATA` or another configured data
root. They are reproduction drivers, not collaborator quickstarts, and must not be
pointed at a read-only release. The commands above always use explicit output roots. Beam's README and core's UVA README document
additional dependencies. `tracker_point` defaults to g4ref_v1 shard 11, uses
the digi/reco/drift paths above, primary-only steps and a fixed-momentum hybrid
F2 fit; it is not the full F1 physics chain. Set `FASTSIMKIT_TRACKER_DATA` to the
release root (not its CRV child) and `FASTSIMKIT_TRACKER_CODE` to the retained
endpoint snapshot. An input publisher must provide a complete manifest
before claiming an end-to-end clean-account setup.
