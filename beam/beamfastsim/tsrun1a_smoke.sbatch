#!/bin/bash
#SBATCH --job-name=ts1_run1a_smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -eo pipefail
: "${TS_SMOKE_OUT:?set a new output directory}"
: "${TS_SMOKE_FCL:?set the raw Run1A FCL path}"
: "${OFFLINE_RUNNER:?set the existing offline_uva.sh path}"
: "${CVMFSEXEC_DIST:?set the existing EL8 cvmfsexec directory}"
mkdir "$TS_SMOKE_OUT"
cd "$TS_SMOKE_OUT"
cp "$TS_SMOKE_FCL" smoke.fcl
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_EVENTS=100 OFFLINE_LOG_PREFIX=smoke OFFLINE_WARM_REPEAT=0
export OFFLINE_OUTPUT_FILE="$PWD/tsrun1a_raw.root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module load apptainer
bash "$OFFLINE_RUNNER" /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac "$PWD/smoke.fcl"
