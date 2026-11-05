#!/bin/bash
# Prepare and digitize each remaining parent once; the first parent's completed artifact is reused.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -eo pipefail
: "${TRIGGER_CODE:?}" "${TRIGGER_CAMPAIGN:?}" "${TRIGGER_ROOT:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
[[ $SLURM_ARRAY_TASK_ID =~ ^[2-9]$ || $SLURM_ARRAY_TASK_ID == 10 ]] || exit 2
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
env_file="$TRIGGER_ROOT/prepared/injection-parent-${SLURM_ARRAY_TASK_ID}-${SLURM_ARRAY_JOB_ID}.env"
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON" "$TRIGGER_CODE/trigger/injection_campaign.py" prepare-parent --plan "$TRIGGER_CAMPAIGN" --ordinal "$SLURM_ARRAY_TASK_ID" --env-output "$env_file"
source "$env_file"
exec bash "$TRIGGER_CODE/trigger/step_inject.sbatch"
