#!/bin/bash
# Round-trip original detector digis through existing plugins without altering released trigger runtime.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_PREPARED:?}"
: "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_PLUGINS:?}"
: "${TRIGGER_INPUT:?}"
: "${TRIGGER_CONFIG:?}"
: "${TRIGGER_AREA:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
: "${CVMFSEXEC_DIST:?}"
: "${OFFLINE_EVENTS:?}"
[[ $OFFLINE_EVENTS =~ ^[1-9][0-9]{0,3}$ ]] || exit 2
[[ ! -e $TRIGGER_OUTPUT && ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
module load apptainer
mkdir -p "$TRIGGER_OUTPUT"
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_OUTPUT,$TRIGGER_PLUGINS,$(dirname "$TRIGGER_INPUT"),$(dirname "$TRIGGER_CONFIG")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/plugin_runtime.py" "$TRIGGER_CODE/container/offline/offline_uva.sh" --plugins "$TRIGGER_PLUGINS" --output "$TRIGGER_OUTPUT/runtime"
cd "$TRIGGER_OUTPUT"
export OFFLINE_PLUGIN_DIR="$TRIGGER_OUTPUT/runtime/plugins"
export OFFLINE_BINDS="$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_OUTPUT,$TRIGGER_PLUGINS,$(dirname "$TRIGGER_INPUT")"
for step in dump inject; do
  export OFFLINE_LOG_PREFIX="$step"
  set +e
  bash "$TRIGGER_OUTPUT/runtime/offline_plugins.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/$step.fcl" > "$step.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$step.exit"
  [[ $status == 0 ]] || { tail -80 "$step.log"; exit "$status"; }
done
"${analysis[@]}" "$TRIGGER_CODE/trigger/event_count.py" combined.art --expected-events "$OFFLINE_EVENTS" --output event-count.json
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_input.py" --reference "$TRIGGER_INPUT" --candidate combined.art --config "$TRIGGER_CONFIG" --expected-events "$OFFLINE_EVENTS" --straw-digi-tag tableSD::R7Adapter --output admission.json
