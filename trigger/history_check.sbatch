#!/bin/bash
# Independently certify saved stock parent history only; never rerun completed production art.
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -eo pipefail
: "${TRIGGER_ROOT:?}" "${TRIGGER_CODE:?}" "${TRIGGER_CONFIG:?}" "${TRIGGER_HISTORY_RECEIPT:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON" "$TRIGGER_CODE/trigger/hybrid_history.py" --config "$TRIGGER_CONFIG" --regeneration-manifest "$TRIGGER_ROOT/prepared/regenerate_v3/manifest.json" --directory "$TRIGGER_ROOT/out/ce_regen_full" --comparison "$TRIGGER_ROOT/out/ce_regen_full/production-comparison-v2.json" --expected-events 28044 --output "$TRIGGER_HISTORY_RECEIPT"
