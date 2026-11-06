#!/bin/bash
#SBATCH --job-name=ts1_run1a_ref
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --array=0-199%32
#SBATCH --no-requeue
set -eo pipefail
: "${TS_CAMPAIGN_ROOT:?set the new campaign directory with frozen config/}"
: "${CVMFSEXEC_DIST:?set the existing EL8 cvmfsexec directory}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array}"
[[ $SLURM_ARRAY_TASK_ID =~ ^(0|[1-9][0-9]*)$ ]] || exit 2
(( SLURM_ARRAY_TASK_ID < 200 )) || exit 2
index=$SLURM_ARRAY_TASK_ID
subrun=$((920000 + index))
seed=$((920001 + 1000 * index))
printf -v shard '%03d' "$index"
printf -v artfile 'sim.user.TSRun1ARaw.campaign_v1.001430_%08d.art' "$subrun"
cohort=development
(( index < 100 )) || cohort=heldout
root=$(cd "$TS_CAMPAIGN_ROOT" && pwd -P)
[[ -f $root/config/tsrun1a_raw.fcl && -f $root/config/offline_uva.sh ]] || exit 2
mkdir "$root/shards/$shard"
cd "$root/shards/$shard"
cp "$root/config/tsrun1a_raw.fcl" shard.fcl
chmod u+w shard.fcl
cat >> shard.fcl <<EOF
source.firstSubRun: $subrun
source.maxEvents: 5000
services.SeedService.baseSeed: $seed
outputs.RawTSOutput.fileName: "$artfile"
EOF
printf 'index\tcohort\trun\tsubrun\tbase_seed\trequested_pot\tjob\n%s\t%s\t1430\t%s\t%s\t5000\t%s\n' \
  "$index" "$cohort" "$subrun" "$seed" "${SLURM_JOB_ID:-unknown}" > shard.tsv
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_EVENTS=5000 OFFLINE_LOG_PREFIX=reference OFFLINE_WARM_REPEAT=0
export OFFLINE_OUTPUT_FILE="$PWD/tsrun1a_raw.root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module load apptainer
bash "$root/config/offline_uva.sh" /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac "$PWD/shard.fcl"
