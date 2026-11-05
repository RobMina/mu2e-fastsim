#!/bin/bash
# Admit all completed parent-boundary injections and execute exactly one all52path trigger history.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -eo pipefail
: "${TRIGGER_CODE:?}" "${TRIGGER_ROOT:?}" "${TRIGGER_CAMPAIGN:?}"
: "${TRIGGER_REGENERATION_MANIFEST:?}" "${TRIGGER_PRODUCTION_COMPARISON:?}" "${TRIGGER_PRODUCTION_LOG:?}"
: "${TRIGGER_STOCK_DIRECTORY:?}" "${TRIGGER_PREPARED:?}" "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_CONFIG:?}" "${TRIGGER_AREA:?}" "${CVMFSEXEC_DIST:?}" "${OFFLINE_IMAGE:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
: "${TRIGGER_EXPECTED_EVENTS:?}"
[[ $TRIGGER_EXPECTED_EVENTS =~ ^[1-9][0-9]{0,5}$ ]] || exit 2
[[ ! -e $TRIGGER_OUTPUT && ! -e $TRIGGER_PREPARED && ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
mkdir -p "$TRIGGER_OUTPUT" "$(dirname "$TRIGGER_PREPARED")"
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
analysis=(apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_campaign.py" --plan "$TRIGGER_CAMPAIGN" --regeneration "$TRIGGER_REGENERATION_MANIFEST" --stock-directory "$TRIGGER_STOCK_DIRECTORY" --comparison "$TRIGGER_PRODUCTION_COMPARISON" --production-log "$TRIGGER_PRODUCTION_LOG" --output "$TRIGGER_PREPARED"
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_replay.py" preflight --config "$TRIGGER_CONFIG" --prepared "$TRIGGER_PREPARED" --expected-events "$TRIGGER_EXPECTED_EVENTS" --runtime-area "$TRIGGER_AREA" --runtime-image "$OFFLINE_IMAGE"
cd "$TRIGGER_OUTPUT"
export OFFLINE_EVENTS="$TRIGGER_EXPECTED_EVENTS" OFFLINE_LOG_PREFIX=replay OFFLINE_BINDS="$binds"
set +e
bash "$TRIGGER_CODE/container/offline/offline_uva.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/replay.fcl" > replay.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -80 replay.log; exit "$status"; }
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_replay.py" validate --config "$TRIGGER_CONFIG" --prepared "$TRIGGER_PREPARED" --art decisions.art --log replay.log --output comparison.json
