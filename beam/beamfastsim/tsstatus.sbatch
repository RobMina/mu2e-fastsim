#!/bin/bash
#SBATCH --job-name=ts_status
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -eo pipefail
case "${TS_STATUS_COHORT-development}" in
  development) indices=({0..99}) ;;
  heldout) indices=({100..199}) ;;
  *) echo 'TS_STATUS_COHORT must be development or heldout' >&2; exit 2 ;;
esac
: "${TS_CAMPAIGN_ROOT:?set completed campaign root}"
: "${TS_GENERATION_JOB:?set generation array job ID for completion logs}"
: "${TS_STATUS_OUT:?set a new analyzer output directory}"
: "${TS_STATUS_FCL:?set stock analyzer-only FCL}"
: "${CVMFSEXEC_DIST:?set existing EL8 client}"
inputs=()
for index in "${indices[@]}"; do
  printf -v shard '%03d' "$index"
  printf -v subrun '%08d' "$((920000 + index))"
  grep -q 'Art has completed and will exit with status 0.' "$TS_CAMPAIGN_ROOT/logs/${TS_GENERATION_JOB}_$index.log"
  inputs+=("$TS_CAMPAIGN_ROOT/shards/$shard/sim.user.TSRun1ARaw.campaign_v1.001430_$subrun.art")
done
mkdir "$TS_STATUS_OUT"
cd "$TS_STATUS_OUT"
cp "$TS_STATUS_FCL" read.fcl
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_WARM_REPEAT=0
export OFFLINE_EVENTS=500000 OFFLINE_LOG_PREFIX=status OFFLINE_OUTPUT_FILE="$PWD/tsstatus.root"
module load apptainer
bash "$TS_CAMPAIGN_ROOT/config/offline_uva.sh" /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac "$PWD/read.fcl" "${inputs[@]}"
