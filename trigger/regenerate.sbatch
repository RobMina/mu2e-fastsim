#!/bin/bash
# Digitize complete original parents with unchanged modules and unfiltered output.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -eo pipefail
: "${TRIGGER_CODE:?set the immutable umbrella snapshot}"
: "${TRIGGER_PREPARED:?set the prepared regeneration}"
: "${TRIGGER_PARENTS:?set the staged parent directory}"
: "${TRIGGER_SOURCE_AUDIT:?set the catalogued source-audit JSON}"
: "${TRIGGER_OUTPUT:?set a new private output directory}"
: "${TRIGGER_AREA:?set the matching already-built Offline release}"
: "${CVMFSEXEC_DIST:?set the existing CVMFS client}"
: "${TRIGGER_ANALYSIS_IMAGE:?set the existing Python analysis image}"
: "${TRIGGER_PYTHON:?set the modern Python interpreter}"
: "${OFFLINE_EVENTS:?set the explicit positive pilot or full-event count}"
[[ $OFFLINE_EVENTS =~ ^[1-9][0-9]{0,8}$ ]] || { echo 'OFFLINE_EVENTS must be bounded positive decimal (at most 9 digits)' >&2; exit 2; }
[[ ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || { echo 'regeneration forbids warm repeats or TFileService overrides' >&2; exit 2; }
module load apptainer
[[ ! -e $TRIGGER_OUTPUT ]] || { echo 'refuse to reuse regeneration output directory' >&2; exit 2; }
mkdir -p "$TRIGGER_OUTPUT"
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$TRIGGER_PREPARED,$TRIGGER_PARENTS,$TRIGGER_OUTPUT,$(dirname "$TRIGGER_SOURCE_AUDIT")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/stage_parents.py" "$TRIGGER_SOURCE_AUDIT" \
  --verify "$TRIGGER_PARENTS" --output "$TRIGGER_OUTPUT/parent-checksums.json"
cd "$TRIGGER_OUTPUT"
export OFFLINE_LOG_PREFIX=regenerate
export OFFLINE_BINDS="${OFFLINE_BINDS:+$OFFLINE_BINDS,}$TRIGGER_PREPARED:$TRIGGER_PREPARED,$TRIGGER_PARENTS:$TRIGGER_PARENTS"
set +e
bash "$TRIGGER_CODE/container/offline/offline_uva.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/regenerate.fcl" > regenerate.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -100 regenerate.log; exit "$status"; }
"${analysis[@]}" "$TRIGGER_CODE/trigger/regeneration_census.py" "$TRIGGER_PREPARED/manifest.json" --directory "$TRIGGER_OUTPUT" --expected-events "$OFFLINE_EVENTS" --output art-entry-count.json
