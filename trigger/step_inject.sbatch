#!/bin/bash
# Digitize primary-only fast steps with released stock electronics while retaining every source event.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_PREPARED:?}"
: "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_PLUGINS:?}"
: "${TRIGGER_SOURCE:?}"
: "${TRIGGER_TRANSPORT:?}"
: "${TRIGGER_CONFIG:?}"
: "${TRIGGER_AREA:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
: "${CVMFSEXEC_DIST:?}"
: "${OFFLINE_EVENTS:?}"
[[ $OFFLINE_EVENTS =~ ^[1-9][0-9]{0,5}$ ]] || exit 2
[[ ! -e $TRIGGER_OUTPUT && ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
[[ -s $TRIGGER_TRANSPORT/step-admission.json ]] || exit 2
module load apptainer
if [[ -n ${TRIGGER_ANALYSIS_SHA256:-} ]]; then
  printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
fi
mkdir -p "$TRIGGER_OUTPUT"
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_OUTPUT,$TRIGGER_PLUGINS,$TRIGGER_TRANSPORT,$(dirname "$TRIGGER_SOURCE"),$(dirname "$TRIGGER_CONFIG")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/step_table_audit.py" --source "$TRIGGER_SOURCE" --states "$TRIGGER_TRANSPORT/transport/states.root" --table "$TRIGGER_TRANSPORT/transport/fasttrk_field.root" --handoff "$TRIGGER_TRANSPORT/transport/handoff.json" --expected-events "$OFFLINE_EVENTS" --output "$TRIGGER_OUTPUT/step-admission.json"
"${analysis[@]}" "$TRIGGER_CODE/trigger/step_handoff.py" "$TRIGGER_CONFIG" --inputs "$TRIGGER_SOURCE" --table "$TRIGGER_TRANSPORT/transport/fasttrk_field.root" --preflight "$TRIGGER_PREPARED" --admission "$TRIGGER_OUTPUT/step-admission.json" --output "$TRIGGER_OUTPUT/execution-binding.json"
"${analysis[@]}" "$TRIGGER_CODE/trigger/plugin_runtime.py" "$TRIGGER_CODE/container/offline/offline_uva.sh" --plugins "$TRIGGER_PLUGINS" --output "$TRIGGER_OUTPUT/runtime"
cd "$TRIGGER_OUTPUT"
export OFFLINE_PLUGIN_DIR="$TRIGGER_OUTPUT/runtime/plugins" OFFLINE_LOG_PREFIX=inject
export OFFLINE_BINDS="$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_OUTPUT,$TRIGGER_PLUGINS,$TRIGGER_TRANSPORT,$(dirname "$TRIGGER_SOURCE")"
set +e
bash "$TRIGGER_OUTPUT/runtime/offline_plugins.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/job.fcl" > inject.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -80 inject.log; exit "$status"; }
"${analysis[@]}" "$TRIGGER_CODE/trigger/event_count.py" unfiltered_1.art --expected-events "$OFFLINE_EVENTS" --output event-count.json
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_input.py" --reference "$TRIGGER_SOURCE" --candidate unfiltered_1.art --config "$TRIGGER_CONFIG" --expected-events "$OFFLINE_EVENTS" --straw-digi-tag R7FastSD::R7TrackerInject --output admission.json
