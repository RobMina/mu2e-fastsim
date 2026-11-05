#!/bin/bash
# Stock calorimeter digi + reco (calo/fcl/stock_chain.fcl) on saved CaloShowerSteps, one CPU, existing runtime only.
#SBATCH --job-name=calo_stock
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:30:00
set -eo pipefail
: "${CALO_CODE:?umbrella snapshot holding calo/ and container/}"
: "${CALO_OUT:?fresh output directory}"
: "${CALO_AREA:?already-built muse area or tarball}"
: "${CALO_INPUT:?art file with CaloShowerSteps}"
: "${CVMFSEXEC_DIST:?prepared cvmfsexec}"
module load apptainer
mkdir -p "$CALO_OUT" && cd "$CALO_OUT"
printf 'job=%s host=%s start=%s input=%s area=%s events=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" \
  "$CALO_INPUT" "$CALO_AREA" "${OFFLINE_EVENTS:-all}" > provenance.txt
export OFFLINE_LOG_PREFIX=stock OFFLINE_OUTPUT_FILE="$PWD/calo_stock.root"
bash "$CALO_CODE/container/offline/offline_uva.sh" "$CALO_AREA" "$CALO_CODE/calo/fcl/${CALO_FCL:-stock_chain.fcl}" "$CALO_INPUT" > stock.log 2>&1
printf 'end=%s\n' "$(date -u +%FT%TZ)" >> provenance.txt
