#!/bin/bash
# Stock calorimeter digi + reco (calo/fcl/stock_chain.fcl) on saved CaloShowerSteps, one CPU, existing runtime only.
#SBATCH --job-name=cal1_stock
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:30:00
set -eo pipefail
: "${CAL1_CODE:?umbrella snapshot holding calo/ and container/}"
: "${CAL1_OUT:?fresh output directory}"
: "${CAL1_AREA:?already-built muse area or tarball}"
: "${CAL1_INPUT:?art file with CaloShowerSteps}"
: "${CVMFSEXEC_DIST:?prepared cvmfsexec}"
module load apptainer
mkdir -p "$CAL1_OUT" && cd "$CAL1_OUT"
printf 'job=%s host=%s start=%s input=%s area=%s events=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" \
  "$CAL1_INPUT" "$CAL1_AREA" "${OFFLINE_EVENTS:-all}" > receipt.txt
export OFFLINE_LOG_PREFIX=stock OFFLINE_OUTPUT_FILE="$PWD/calo_stock.root"
bash "$CAL1_CODE/container/offline/offline_uva.sh" "$CAL1_AREA" "$CAL1_CODE/calo/fcl/${CAL1_FCL:-stock_chain.fcl}" "$CAL1_INPUT" > stock.log 2>&1
printf 'end=%s\n' "$(date -u +%FT%TZ)" >> receipt.txt
