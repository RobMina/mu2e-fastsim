#!/bin/bash
# Validate the completed unfiltered regeneration before preparing a second pass.
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:10:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_PREPARED:?}"
: "${TRIGGER_PARENTS:?}"
: "${TRIGGER_REGENERATION:?}"
: "${TRIGGER_REFERENCE:?}"
: "${TRIGGER_CONFIG:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
: "${TRIGGER_EXPECTED_EVENTS:?}"
[[ $TRIGGER_EXPECTED_EVENTS =~ ^[1-9][0-9]{0,8}$ ]] || exit 2
[[ $(cat "$TRIGGER_REGENERATION/art.exit") == 0 ]] || { echo 'regeneration did not finish art successfully' >&2; exit 2; }
report=${TRIGGER_REPORT:-$TRIGGER_REGENERATION/production-comparison.json}
[[ ! -e $report ]] || { echo 'refuse comparison report overwrite' >&2; exit 2; }
module load apptainer
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_PARENTS,$TRIGGER_REGENERATION,$(dirname "$TRIGGER_REFERENCE"),$(dirname "$TRIGGER_CONFIG")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/regenerated_compare.py" "$TRIGGER_REFERENCE" "$TRIGGER_REGENERATION" \
  --regeneration-manifest "$TRIGGER_PREPARED/manifest.json" --config "$TRIGGER_CONFIG" \
  --expected-events "$TRIGGER_EXPECTED_EVENTS" --require-all-reference --output "$report"
