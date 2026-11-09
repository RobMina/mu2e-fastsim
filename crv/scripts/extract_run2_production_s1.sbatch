#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
# Convert one already-staged S1 file with existing plugins; never run Geant4 or request more inputs.
set -euo pipefail
: "${REUSE_CODE:?}" "${REUSE_ROOT:?}" "${PRODUCTION_INPUT:?}" "${PRODUCTION_SHA256:?}"
: "${OFFLINE_RUNNER:?}" "${OFFLINE_AREA:?}" "${CVMFSEXEC_DIST:?}"
module load apptainer
[[ -f "$PRODUCTION_INPUT" ]] || { echo 'stage the single verified production input first'; exit 2; }
printf '%s  %s\n' "$PRODUCTION_SHA256" "$PRODUCTION_INPUT" | sha256sum -c -
OUT="$REUSE_ROOT/extract_${SLURM_JOB_ID:?}"
mkdir "$OUT"
cd "$OUT"
export OFFLINE_EVENTS=1 OFFLINE_OUTPUT_FILE="$OUT/probe.root" OFFLINE_LOG_PREFIX=probe
bash "$OFFLINE_RUNNER" "$OFFLINE_AREA" "$REUSE_CODE/art/CrvFastSim/fcl/probe_run2_s1_products.fcl" "$PRODUCTION_INPUT" > probe.log 2>&1
unset OFFLINE_EVENTS
export OFFLINE_OUTPUT_FILE="$OUT/s1steps.root" OFFLINE_LOG_PREFIX=full
bash "$OFFLINE_RUNNER" "$OFFLINE_AREA" "$REUSE_CODE/art/CrvFastSim/fcl/dump_run2_s1.fcl" "$PRODUCTION_INPUT" > full.log 2>&1
