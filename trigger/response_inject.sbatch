#!/bin/bash
# Digitize one admitted full-stock parent once, then read-only re-export SGS for exact numerical roundtrip.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -eo pipefail
: "${TRIGGER_CODE:?}" "${TRIGGER_ROOT:?}" "${TRIGGER_CONTROL_PLAN:?}" "${TRIGGER_OUTPUT_ROOT:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
[[ $SLURM_ARRAY_TASK_ID =~ ^[1-9]$ || $SLURM_ARRAY_TASK_ID == 10 ]] || exit 2
[[ ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
analysis=(apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
output="$TRIGGER_OUTPUT_ROOT/parent_$SLURM_ARRAY_TASK_ID"
[[ ! -e $output ]] || exit 2
mkdir -p "$output"
"${analysis[@]}" "$TRIGGER_CODE/trigger/response_launch.py" --plan "$TRIGGER_CONTROL_PLAN" \
  --ordinal "$SLURM_ARRAY_TASK_ID" --output "$output"
source "$output/launch.env"
cd "$output"
export OFFLINE_EVENTS OFFLINE_IMAGE CVMFSEXEC_DIST
export OFFLINE_PLUGIN_DIR="$CONTROL_PLUGINS" OFFLINE_BINDS="$binds" OFFLINE_LOG_PREFIX=inject
set +e
bash "$CONTROL_RUNNER" "$TRIGGER_AREA" "$CONTROL_PREPARED/job.fcl" > inject.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -80 inject.log; exit "$status"; }
"${analysis[@]}" "$TRIGGER_CODE/trigger/response_control.py" prepare-dump --config "$CONTROL_CONFIG" \
  --candidate "$output/unfiltered_1.art" --output "$output/dump-prepared"
cd "$output/roundtrip"
export OFFLINE_LOG_PREFIX=dump
set +e
bash "$CONTROL_RUNNER" "$TRIGGER_AREA" "$output/dump-prepared/job.fcl" > dump.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -80 dump.log; exit "$status"; }
"${analysis[@]}" "$TRIGGER_CODE/trigger/response_validate.py" --plan "$TRIGGER_CONTROL_PLAN" \
  --ordinal "$SLURM_ARRAY_TASK_ID" --directory "$output" --output "$output/completed.json"
