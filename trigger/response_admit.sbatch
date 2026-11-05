#!/bin/bash
# Admit full-stock tables and source collection presence without running transport, digitization, or trigger.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -eo pipefail
: "${TRIGGER_CODE:?}" "${TRIGGER_ROOT:?}" "${TRIGGER_FAST_PLAN:?}" "${TRIGGER_TABLE_RECEIPT:?}"
: "${TRIGGER_FIRST_TABLE:?}" "${TRIGGER_REMAINING_TABLE:?}" "${TRIGGER_PREPARED:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
: "${TRIGGER_FAST_PREPARED:?}" "${TRIGGER_FAST_ART:?}" "${TRIGGER_FAST_COMPARISON:?}"
[[ ! -e $TRIGGER_PREPARED ]] || exit 2
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON" \
  "$TRIGGER_CODE/trigger/response_control.py" create --fast-plan "$TRIGGER_FAST_PLAN" \
  --tables "$TRIGGER_FIRST_TABLE" "$TRIGGER_REMAINING_TABLE" --table-receipt "$TRIGGER_TABLE_RECEIPT" \
  --output "$TRIGGER_PREPARED" --fast-prepared "$TRIGGER_FAST_PREPARED" \
  --fast-art "$TRIGGER_FAST_ART" --fast-comparison "$TRIGGER_FAST_COMPARISON"
