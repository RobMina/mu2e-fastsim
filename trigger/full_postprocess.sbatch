#!/bin/bash
# Validate saved full-history art without rerunning simulation or reconstruction.
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:10:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_PREPARED:?}"
: "${TRIGGER_REGENERATION:?}"
: "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_CONFIG:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
: "${TRIGGER_REPORT:?}"
[[ $(cat "$TRIGGER_OUTPUT/art.exit") == 0 ]] || { echo 'saved art did not finish successfully' >&2; exit 2; }
[[ ! -e $TRIGGER_REPORT ]] || { echo 'refuse comparison report overwrite' >&2; exit 2; }
module load apptainer
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_REGENERATION,$TRIGGER_OUTPUT,$(dirname "$TRIGGER_CONFIG"),$(dirname "$TRIGGER_REPORT")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/full_validate.py" "$TRIGGER_OUTPUT/decisions.art" \
  --prepared "$TRIGGER_PREPARED" --directory "$TRIGGER_REGENERATION" --config "$TRIGGER_CONFIG" \
  --log "$TRIGGER_OUTPUT/replay.log" --output "$TRIGGER_REPORT"
