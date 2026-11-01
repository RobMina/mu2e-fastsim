# CRV: running the default chain

Work from a clone of the umbrella repository [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim), which pins
compatible commits of core, crv and tracker. Its [`docs/running.md`](https://github.com/RobMina/mu2e-fastsim/blob/main/docs/running.md)
covers the environment, `site.env` and the input release; this page holds the CRV commands.

## Minimum site configuration

```bash
export M2FS_CODE=/path/to/mu2e-fastsim                 # the umbrella clone
export M2FS_INPUTS=/path/to/verified/input-release     # read-only
export M2FS_DATA=/path/to/writable/outputs
export PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/crv"
export CRVFS_DATA="$M2FS_INPUTS/crv_fastsim"
export CRVFS_LOOKUP_DIR="$CRVFS_DATA/lookup_v6_0"       # photon lookup tables and photonMap.root
export MU2E_SEARCH_PATH="$M2FS_INPUTS/offline"         # Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt, singlePEWaveform_v3.txt
```

Python 3.10+ with numpy, pandas, uproot, awkward, scipy, matplotlib, pyarrow and PyYAML (`pip install -e core -e crv`
from the umbrella clone, or the project's Apptainer image); JAX only for the device options.

## One shard of the default chain (CPU)

Default = extracted_v04, `range_out` transport, tier-1b set `strict_v4`, NumPy everywhere.

```bash
G="$CRVFS_DATA/g4ref_v2_tables/gen.001400_00000022.root"     # 50,000 generated CRY events
GEOM="$CRVFS_DATA/geometry"
python3 -m crvfastsim runchain "$G" --tier1b-dir "$CRVFS_DATA/tier1b_strict_v2" \
  --gdml "$GEOM/mu2e_extracted_v04.gdml" --calib "$GEOM/crvcalib_sim_best_v1_5_run1400.csv" \
  --deviation "$GEOM/crvphoton_sim_best_v1_5_run1400.csv" \
  --max-events 1000 --seed 22 --outdir "$M2FS_DATA/crv-smoke/chain"
```

`runchain_stats.json` in the output directory holds the counts per stage (steps, digis, pulses, clusters per sector
type) and the cost per stage; the resolved tier-1b set, its pinned hash and every input path are recorded there too.
Remove `--max-events` for the whole shard (a few minutes on one core; peak memory about 9 GB). The `strict_v4` files
live in the directory `tier1b_strict_v2` (the set reuses that library).

Timing, fingerprints and repeated passes on the same inputs:

```bash
python3 -m crvfastsim e2e-bench "$G" --tier1b-dir "$CRVFS_DATA/tier1b_strict_v2" \
  --gdml "$GEOM/mu2e_extracted_v04.gdml" --calib "$GEOM/crvcalib_sim_best_v1_5_run1400.csv" \
  --deviation "$GEOM/crvphoton_sim_best_v1_5_run1400.csv" \
  --max-events 1000 --seed 22 --repeat 2 --outdir "$M2FS_DATA/crv-smoke/bench"
```

Two passes with the same seed must give identical fingerprints.

## On a GPU

Add `--jax` (response, waveforms, reco on the device) and optionally `--transport-backend jax --tier1b-backend jax`
([device.md](device.md); the device tier 1b is a speed option, see [limitations.md](limitations.md)). Set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` on a shared card. Device precision is float64.

## Options that change the physics

| Option | Effect |
|---|---|
| `--tier1b NAME` | another tier-1b set (`strict_v3`, `strict_v2`, `legacy`; see [tier1b-default.md](tier1b-default.md)) |
| `--library LIB` | a custom tier-1b library without placement |
| `--legacy-transport` | the stage-C transport before `range_out` ([transport-range-out.md](transport-range-out.md)) |
| `--run-config NAME` | a named configuration (needs a tier-1b library that does not exist today) |
| `--gen native|device --gen-events N` | CRY primaries generated in process instead of gen tables ([native-cry.md](native-cry.md)) |

## Batch work

Analyses that take many files support `--emit batch_NN.json` on a slice of files and `--merge "batches/*.json"`;
merged counts are sums, so a merge equals a single pass. `runchain-merge` combines `runchain --emit` files of a node
job.

## Endpoints

The Mu2eBO kit (`fastsimkit`) runs one CRV point as one job on a Slurm cluster (`chain_point`, inputs streamed
into node memory) or on Fermilab EAF (`chain`, inputs read in place). Its configuration is in the umbrella's
`site.env.example` (`FASTSIMKIT_*`); core's README describes the kit.
