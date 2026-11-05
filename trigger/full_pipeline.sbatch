#!/bin/bash
# Prepare only after a successful full production comparison, then run one full-history replay.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_PREPARED:?}"
: "${TRIGGER_REGENERATION_PREPARED:?}"
: "${TRIGGER_REGENERATION:?}"
: "${TRIGGER_PARENTS:?}"
: "${TRIGGER_REFERENCE:?}"
: "${TRIGGER_CONFIG:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
: "${TRIGGER_EXPECTED_EVENTS:?}"
[[ $TRIGGER_EXPECTED_EVENTS =~ ^[1-9][0-9]{0,8}$ ]] || exit 2
[[ ! -e $TRIGGER_PREPARED ]] || { echo 'refuse preparation reuse' >&2; exit 2; }
[[ ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
module load apptainer
mkdir -p "$(dirname "$TRIGGER_PREPARED")"
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_REGENERATION_PREPARED,$TRIGGER_REGENERATION,$TRIGGER_PARENTS,$(dirname "$TRIGGER_PREPARED"),$(dirname "$TRIGGER_REFERENCE"),$(dirname "$TRIGGER_CONFIG")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/full_replay.py" "$TRIGGER_CONFIG" \
  --regeneration-manifest "$TRIGGER_REGENERATION_PREPARED/manifest.json" --directory "$TRIGGER_REGENERATION" \
  --production-comparison "$TRIGGER_REGENERATION/production-comparison.json" \
  --expected-events "$TRIGGER_EXPECTED_EVENTS" --output "$TRIGGER_PREPARED"
exec bash "$TRIGGER_CODE/trigger/full_replay.sbatch"
