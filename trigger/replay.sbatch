#!/bin/bash
# Run a bounded stock-digi replay with the existing Offline container bridge.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -eo pipefail
: "${TRIGGER_CODE:?set the umbrella snapshot}"
: "${TRIGGER_PREPARED:?set the prepared configuration directory}"
: "${TRIGGER_INPUT:?set the staged stock digi file}"
: "${TRIGGER_OUTPUT:?set an output directory}"
: "${TRIGGER_AREA:?set the matching already-built Offline release}"
: "${CVMFSEXEC_DIST:?set the existing CVMFS client}"
: "${TRIGGER_PYTHON:?set a Python 3.10+ interpreter or use TRIGGER_ANALYSIS_IMAGE}"
module load apptainer
mkdir -p "$TRIGGER_OUTPUT"
analysis_python=("$TRIGGER_PYTHON")
if [[ -n ${TRIGGER_ANALYSIS_IMAGE:-} ]]; then
  analysis_binds="$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_OUTPUT"
  if [[ -n ${TRIGGER_EVENT_LIST:-} ]]; then
    analysis_binds+=",$(dirname "$TRIGGER_EVENT_LIST")"
  fi
  analysis_python=(apptainer exec --cleanenv -B "$analysis_binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
fi
[[ ${TRIGGER_FORCE:-0} == 1 || ( ! -e $TRIGGER_OUTPUT/art.exit && ! -e $TRIGGER_OUTPUT/replay.log ) ]] ||
  { echo 'refuse to overwrite a previous replay (set TRIGGER_FORCE=1)' >&2; exit 2; }
"${analysis_python[@]}" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
cd "$TRIGGER_OUTPUT"
# OFFLINE_EVENTS=0 replays the whole file; compare.py then takes the count from TRIGGER_EVENT_LIST.
expected=${OFFLINE_EVENTS:-100}
if (( expected > 0 )); then export OFFLINE_EVENTS=$expected; else unset OFFLINE_EVENTS; expected=0; fi
export OFFLINE_LOG_PREFIX=trigger
export OFFLINE_BINDS="${OFFLINE_BINDS:+$OFFLINE_BINDS,}$TRIGGER_PREPARED:$TRIGGER_PREPARED"
set +e
bash "$TRIGGER_CODE/container/offline/offline_uva.sh" "$TRIGGER_AREA" \
  "$TRIGGER_PREPARED/replay.fcl" "$TRIGGER_INPUT" > replay.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -100 replay.log; exit "$status"; }
if [[ ${TRIGGER_MODE:-decisions} == timing ]]; then
  "${analysis_python[@]}" "$TRIGGER_CODE/trigger/timing.py" replay.log \
    --inventory "$TRIGGER_PREPARED/replay_inventory.json" --output timing.json --top 10 ${OFFLINE_EVENTS:+--expected-events "$OFFLINE_EVENTS"}
  exit 0
fi
comparison_args=(--manifest "$TRIGGER_PREPARED/manifest.json" --expected-events "$expected" --output comparison.json)
if [[ -n ${TRIGGER_EVENT_LIST:-} ]]; then
  comparison_args+=(--input-event-list "$TRIGGER_EVENT_LIST")
fi
"${analysis_python[@]}" "$TRIGGER_CODE/trigger/compare.py" replay.log "${comparison_args[@]}"
