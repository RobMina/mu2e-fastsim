#!/bin/bash
# Run an explicitly limited startup probe on a catalogue-checked cached first parent, not a full control.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_PREPARED:?}"
: "${TRIGGER_PARENTS:?}"
: "${TRIGGER_SOURCE_AUDIT:?}"
: "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_AREA:?}"
: "${CVMFSEXEC_DIST:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
: "${OFFLINE_EVENTS:?}"
[[ $OFFLINE_EVENTS =~ ^[1-9][0-9]{0,3}$ ]] || exit 2
[[ ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
[[ ! -e $TRIGGER_OUTPUT ]] || { echo 'refuse prefix probe output reuse' >&2; exit 2; }
module load apptainer
mkdir -p "$TRIGGER_OUTPUT"
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_PARENTS,$TRIGGER_OUTPUT,$(dirname "$TRIGGER_SOURCE_AUDIT")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/catalog_bytes.py" "$TRIGGER_SOURCE_AUDIT" \
  --directory "$TRIGGER_PARENTS" --parent-count 1 --output "$TRIGGER_OUTPUT/first-parent-catalog-checksums.json"
cd "$TRIGGER_OUTPUT"
printf '%s\n' 'LIMITED STARTUP PROBE: source SHA receipt unavailable; full source and production overlap NOT certified' > scope.txt
export OFFLINE_LOG_PREFIX=prefix
export OFFLINE_BINDS="${OFFLINE_BINDS:+$OFFLINE_BINDS,}$TRIGGER_PREPARED:$TRIGGER_PREPARED,$TRIGGER_PARENTS:$TRIGGER_PARENTS"
set +e
bash "$TRIGGER_CODE/container/offline/offline_uva.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/regenerate.fcl" > prefix.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -100 prefix.log; exit "$status"; }
"${analysis[@]}" "$TRIGGER_CODE/trigger/event_count.py" unfiltered_1.art --expected-events "$OFFLINE_EVENTS" --output event-count.json
"${analysis[@]}" "$TRIGGER_CODE/trigger/serialized.py" unfiltered_1.art --output serialized-probe.json
