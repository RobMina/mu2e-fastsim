#!/bin/bash
# Measure every scheduled trigger path on admitted replacement digis; migrations are results, not failures.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_PREPARED:?}"
: "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_INJECTION:?}"
: "${TRIGGER_SOURCE:?}"
: "${TRIGGER_CONFIG:?}"
: "${TRIGGER_AREA:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
: "${TRIGGER_EXPECTED_EVENTS:?}"
: "${CVMFSEXEC_DIST:?}"
[[ $TRIGGER_EXPECTED_EVENTS =~ ^[1-9][0-9]{0,5}$ ]] || exit 2
[[ ! -e $TRIGGER_OUTPUT && ! -e $TRIGGER_PREPARED && ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
combined=${TRIGGER_COMBINED_FILE:-unfiltered_1.art}
completion=${TRIGGER_INJECTION_EXIT:-art.exit}
[[ $combined =~ ^[A-Za-z_][A-Za-z_0-9]*\.art$ && $completion =~ ^[a-z]+\.exit$ ]] || exit 2
[[ $(cat "$TRIGGER_INJECTION/$completion") == 0 ]] || exit 2
module load apptainer
mkdir -p "$TRIGGER_OUTPUT" "$(dirname "$TRIGGER_PREPARED")"
analysis=(apptainer exec --cleanenv -B "$TRIGGER_CODE,$(dirname "$TRIGGER_PREPARED"),$TRIGGER_OUTPUT,$TRIGGER_INJECTION,$(dirname "$TRIGGER_SOURCE"),$(dirname "$TRIGGER_CONFIG")" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_replay.py" prepare --config "$TRIGGER_CONFIG" --prepared "$TRIGGER_PREPARED" --reference "$TRIGGER_SOURCE" --candidate "$TRIGGER_INJECTION/$combined" --admission "$TRIGGER_INJECTION/admission.json" --expected-events "$TRIGGER_EXPECTED_EVENTS"
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_replay.py" preflight --config "$TRIGGER_CONFIG" --prepared "$TRIGGER_PREPARED"
cd "$TRIGGER_OUTPUT"
export OFFLINE_EVENTS="$TRIGGER_EXPECTED_EVENTS" OFFLINE_LOG_PREFIX=replay
export OFFLINE_BINDS="$TRIGGER_PREPARED,$TRIGGER_INJECTION"
set +e
bash "$TRIGGER_CODE/container/offline/offline_uva.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/replay.fcl" > replay.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -80 replay.log; exit "$status"; }
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_replay.py" validate --config "$TRIGGER_CONFIG" --prepared "$TRIGGER_PREPARED" --art decisions.art --log replay.log --output comparison.json
