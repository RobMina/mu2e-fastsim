#!/bin/bash
#SBATCH --job-name=ts1_heldout_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=100-199%16
#SBATCH --no-requeue
set -eo pipefail
: "${TS_CAMPAIGN_ROOT:?set existing campaign root}"
: "${TS_GENERATION_JOB:?set original generation job ID}"
: "${TS_HELDOUT_OUT:?set new held-out extraction directory}"
: "${TS_BEAM_CODE:?set existing beam package directory}"
: "${TS_ANALYSIS_IMAGE:?set existing uproot analysis image}"
: "${TS_ANALYSIS_BIND:?set container binds covering campaign, output and beam code}"
: "${SLURM_ARRAY_TASK_ID:?submit explicit held-out array}"
[[ $SLURM_ARRAY_TASK_ID =~ ^(0|[1-9][0-9]*)$ ]] || exit 2
(( SLURM_ARRAY_TASK_ID >= 100 && SLURM_ARRAY_TASK_ID < 200 )) || exit 2
printf -v shard '%03d' "$SLURM_ARRAY_TASK_ID"
[[ ! -e $TS_HELDOUT_OUT/census-$shard.json && ! -e $TS_HELDOUT_OUT/reference-$shard.npz ]] || exit 2
export PYTHONPATH="$TS_BEAM_CODE" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module load apptainer/1.4.5
apptainer exec --bind "$TS_ANALYSIS_BIND" "$TS_ANALYSIS_IMAGE" python3 -m beamfastsim.tscampaign \
  --root "$TS_CAMPAIGN_ROOT" --index "$SLURM_ARRAY_TASK_ID" --cohort heldout \
  --generation-job "$TS_GENERATION_JOB" --out "$TS_HELDOUT_OUT/census-$shard.json"
apptainer exec --bind "$TS_ANALYSIS_BIND" "$TS_ANALYSIS_IMAGE" python3 -m beamfastsim.tsrawreference \
  --ntuple "$TS_CAMPAIGN_ROOT/shards/$shard/tsrun1a_raw.root" \
  --out "$TS_HELDOUT_OUT/reference-$shard.npz"
