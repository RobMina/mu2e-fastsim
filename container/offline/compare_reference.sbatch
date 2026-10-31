#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=offline_stock_compare
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -eo pipefail
ROOT=${OFFLINE_ROOT:?set OFFLINE_ROOT}
INPUT=${OFFLINE_INPUT:?set OFFLINE_INPUT}
BRIDGE=${OFFLINE_BRIDGE:?set OFFLINE_BRIDGE to the completed bridge directory}
module load apptainer
for arm in ${OFFLINE_BRIDGE_ARMS:-a0 a0g}; do
  apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
    -B "$ROOT:$ROOT" -B "$INPUT:$INPUT" "${FASTSIM_IMAGE:?set FASTSIM_IMAGE}" \
    python3 "$ROOT/code/container/offline/compare_root.py" \
    "${OFFLINE_REFERENCE:?set OFFLINE_REFERENCE to the reference DQM directory}/$arm/valhist.${OFFLINE_SHARD:-001400_00000022}.root" \
    "$BRIDGE/$arm.root" --output "$BRIDGE/$arm.comparison.json"
  if [[ ${OFFLINE_COMPARE_WARM:-0} == 1 ]]; then
    apptainer exec --cleanenv --env OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
      -B "$ROOT:$ROOT" "${FASTSIM_IMAGE}" \
      python3 "$ROOT/code/container/offline/compare_root.py" "$BRIDGE/$arm.root" \
      "$BRIDGE/$arm.warm.root" --output "$BRIDGE/$arm.warm.comparison.json"
  fi
done
