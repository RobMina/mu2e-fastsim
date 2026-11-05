#!/bin/bash
# Require all separately admitted full-stock response parents and run one all52 trigger history.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -eo pipefail
: "${TRIGGER_CODE:?}" "${TRIGGER_ROOT:?}" "${TRIGGER_CONTROL_PLAN:?}" "${TRIGGER_OUTPUT_ROOT:?}"
: "${TRIGGER_FAST_PREPARED:?}" "${TRIGGER_PREPARED:?}" "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_CONFIG:?}" "${TRIGGER_AREA:?}" "${CVMFSEXEC_DIST:?}" "${OFFLINE_IMAGE:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
[[ ! -e $TRIGGER_OUTPUT && ! -e $TRIGGER_PREPARED && ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
mkdir -p "$TRIGGER_OUTPUT"
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
analysis=(apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/response_campaign.py" prepare --plan "$TRIGGER_CONTROL_PLAN" \
  --output-root "$TRIGGER_OUTPUT_ROOT" --fast-prepared "$TRIGGER_FAST_PREPARED" --output "$TRIGGER_PREPARED"
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_replay.py" preflight --config "$TRIGGER_CONFIG" \
  --prepared "$TRIGGER_PREPARED" --expected-events 28044 --runtime-area "$TRIGGER_AREA" --runtime-image "$OFFLINE_IMAGE"
cd "$TRIGGER_OUTPUT"
export OFFLINE_EVENTS=28044 OFFLINE_LOG_PREFIX=replay OFFLINE_BINDS="$binds"
set +e
bash "$TRIGGER_CODE/container/offline/offline_uva.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/replay.fcl" > replay.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -80 replay.log; exit "$status"; }
"${analysis[@]}" "$TRIGGER_CODE/trigger/response_campaign.py" validate --config "$TRIGGER_CONFIG" \
  --prepared "$TRIGGER_PREPARED" --art decisions.art --log replay.log --output comparison.json
