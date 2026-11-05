#!/bin/bash
# Measure separate stock/fast quiet histories sequentially in one CPU allocation.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -eo pipefail
: "${TRIGGER_ROOT:?}" "${TRIGGER_CODE:?}" "${TRIGGER_CONFIG:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
: "${TRIGGER_AREA:?}" "${OFFLINE_IMAGE:?}" "${CVMFSEXEC_DIST:?}"
: "${TRIGGER_QUIET_ROOT:?}" "${TRIGGER_EXPECTED_EVENTS:?}"
: "${TRIGGER_STOCK_PREPARED:?}" "${TRIGGER_STOCK_ART:?}" "${TRIGGER_STOCK_COMPARISON:?}"
: "${TRIGGER_FAST_PREPARED:?}" "${TRIGGER_FAST_ART:?}" "${TRIGGER_FAST_COMPARISON:?}"
[[ $TRIGGER_EXPECTED_EVENTS =~ ^[1-9][0-9]{0,5}$ && ! -e $TRIGGER_QUIET_ROOT ]] || exit 2
[[ ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
mkdir -p "$TRIGGER_QUIET_ROOT"
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
analysis=(apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
printf 'host=%s job=%s order=stock,fast\n' "$(hostname)" "$SLURM_JOB_ID" > "$TRIGGER_QUIET_ROOT/allocation.txt"
for arm in stock fast; do
  if [[ $arm == stock ]]; then
    base_prepared=$TRIGGER_STOCK_PREPARED; base_art=$TRIGGER_STOCK_ART; base_comparison=$TRIGGER_STOCK_COMPARISON
  else
    base_prepared=$TRIGGER_FAST_PREPARED; base_art=$TRIGGER_FAST_ART; base_comparison=$TRIGGER_FAST_COMPARISON
  fi
  prepared=$TRIGGER_QUIET_ROOT/$arm/prepared
  output=$TRIGGER_QUIET_ROOT/$arm/output
  mkdir -p "$output"
  "${analysis[@]}" "$TRIGGER_CODE/trigger/quiet_replay.py" prepare --config "$TRIGGER_CONFIG" --baseline-prepared "$base_prepared" --baseline-art "$base_art" --baseline-comparison "$base_comparison" --prepared "$prepared" --expected-events "$TRIGGER_EXPECTED_EVENTS" --runtime-area "$TRIGGER_AREA" --runtime-image "$OFFLINE_IMAGE"
  "${analysis[@]}" "$TRIGGER_CODE/trigger/quiet_replay.py" preflight --config "$TRIGGER_CONFIG" --prepared "$prepared" --expected-events "$TRIGGER_EXPECTED_EVENTS" --runtime-area "$TRIGGER_AREA" --runtime-image "$OFFLINE_IMAGE"
  cd "$output"
  export OFFLINE_EVENTS="$TRIGGER_EXPECTED_EVENTS" OFFLINE_LOG_PREFIX=replay OFFLINE_BINDS="$binds"
  set +e
  bash "$TRIGGER_CODE/container/offline/offline_uva.sh" "$TRIGGER_AREA" "$prepared/replay.fcl" > replay.log 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > art.exit
  [[ $status == 0 ]] || { tail -80 replay.log; exit "$status"; }
  "${analysis[@]}" "$TRIGGER_CODE/trigger/quiet_replay.py" validate --config "$TRIGGER_CONFIG" --prepared "$prepared" --art decisions.art --log replay.log --output comparison.json --expected-events "$TRIGGER_EXPECTED_EVENTS" --runtime-area "$TRIGGER_AREA" --runtime-image "$OFFLINE_IMAGE"
  printf '%s completed\n' "$arm" >> "$TRIGGER_QUIET_ROOT/allocation.txt"
done
