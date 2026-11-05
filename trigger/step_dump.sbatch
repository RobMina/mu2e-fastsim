#!/bin/bash
# Export existing CE parent truth with existing plugins and capture the released geometry source.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_PREPARED:?}"
: "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_PLUGINS:?}"
: "${TRIGGER_INPUT_DIR:?}"
: "${TRIGGER_AREA:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
: "${CVMFSEXEC_DIST:?}"
: "${OFFLINE_EVENTS:?}"
[[ $OFFLINE_EVENTS =~ ^[1-9][0-9]{0,5}$ ]] || exit 2
[[ ! -e $TRIGGER_OUTPUT && ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
module load apptainer
mkdir -p "$TRIGGER_OUTPUT"
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_OUTPUT,$TRIGGER_PLUGINS" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/plugin_runtime.py" "$TRIGGER_CODE/container/offline/offline_uva.sh" --plugins "$TRIGGER_PLUGINS" --output "$TRIGGER_OUTPUT/runtime" --capture-geometry
cd "$TRIGGER_OUTPUT"
export OFFLINE_PLUGIN_DIR="$TRIGGER_OUTPUT/runtime/plugins" OFFLINE_LOG_PREFIX=dump
export OFFLINE_BINDS="$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_OUTPUT,$TRIGGER_PLUGINS,$TRIGGER_INPUT_DIR"
set +e
bash "$TRIGGER_OUTPUT/runtime/offline_plugins.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/job.fcl" > dump.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -80 dump.log; exit "$status"; }
