#!/bin/bash
# Diagnose saved first-parent decisions only; never rerun art or transport.
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -eo pipefail
: "${TRIGGER_ROOT:?}" "${TRIGGER_CODE:?}" "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
apptainer exec --cleanenv -B "$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON" "$TRIGGER_CODE/trigger/migration_endpoints.py" --art "$TRIGGER_ROOT/out/fast_trigger_parent1_v1/decisions.art" --config "$TRIGGER_ROOT/provenance/ce_ba55.regeneration-source.txt" --prepared "$TRIGGER_ROOT/prepared/fast_trigger_parent1_v1" --comparison "$TRIGGER_ROOT/out/fast_trigger_parent1_v1/comparison.json" --output "$TRIGGER_ROOT/out/fast_trigger_parent1_v1/migration-endpoints-v1.json"
