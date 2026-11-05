#!/bin/bash
# Partition saved decisions without restarting art or any filter counters.
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -eo pipefail
: "${TRIGGER_ROOT:?}" "${TRIGGER_CODE:?}" "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
: "${TRIGGER_DIAGNOSTIC_ART:?}" "${TRIGGER_CONFIG:?}" "${TRIGGER_PREPARED:?}"
: "${TRIGGER_COMPARISON:?}" "${TRIGGER_DIAGNOSTIC_OUTPUT:?}"
[[ ! -e $TRIGGER_DIAGNOSTIC_OUTPUT ]] || exit 2
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
apptainer exec --cleanenv -B "$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON" "$TRIGGER_CODE/trigger/migration_partitions.py" --art "$TRIGGER_DIAGNOSTIC_ART" --config "$TRIGGER_CONFIG" --prepared "$TRIGGER_PREPARED" --comparison "$TRIGGER_COMPARISON" --output "$TRIGGER_DIAGNOSTIC_OUTPUT"
