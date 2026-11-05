#!/bin/bash
# Replay all regenerated parent-boundary inputs without restarting trigger modules.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_PREPARED:?}"
: "${TRIGGER_REGENERATION:?}"
: "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_AREA:?}"
: "${CVMFSEXEC_DIST:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
: "${TRIGGER_CONFIG:?}"
: "${TRIGGER_EXPECTED_EVENTS:?}"
[[ $TRIGGER_EXPECTED_EVENTS =~ ^[1-9][0-9]{0,8}$ ]] || exit 2
[[ ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
[[ ! -e $TRIGGER_OUTPUT ]] || { echo 'refuse output reuse' >&2; exit 2; }
module load apptainer
mkdir -p "$TRIGGER_OUTPUT"
cd "$TRIGGER_OUTPUT"
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_REGENERATION,$TRIGGER_OUTPUT,$(dirname "$TRIGGER_CONFIG")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/full_preflight.py" --prepared "$TRIGGER_PREPARED" --directory "$TRIGGER_REGENERATION" --config "$TRIGGER_CONFIG" --expected-events "$TRIGGER_EXPECTED_EVENTS"
export OFFLINE_EVENTS="$TRIGGER_EXPECTED_EVENTS" OFFLINE_LOG_PREFIX=fullreplay
export OFFLINE_BINDS="${OFFLINE_BINDS:+$OFFLINE_BINDS,}$TRIGGER_REGENERATION:$TRIGGER_REGENERATION,$TRIGGER_PREPARED:$TRIGGER_PREPARED"
set +e
bash "$TRIGGER_CODE/container/offline/offline_uva.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/replay.fcl" > replay.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -100 replay.log; exit "$status"; }
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_REGENERATION,$TRIGGER_OUTPUT,$(dirname "$TRIGGER_CONFIG")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/full_validate.py" decisions.art --prepared "$TRIGGER_PREPARED" --directory "$TRIGGER_REGENERATION" --config "$TRIGGER_CONFIG" --log replay.log --output comparison.json
