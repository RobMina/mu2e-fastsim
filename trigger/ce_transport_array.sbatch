#!/bin/bash
# One frozen primary-only transport per remaining original parent, with bounded array concurrency at submission.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:30:00
set -eo pipefail
: "${TRIGGER_ROOT:?}"
: "${TRIGGER_CODE:?}"
: "${TRIGGER_SPLIT:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
ordinal=$SLURM_ARRAY_TASK_ID
[[ $ordinal =~ ^[2-9]$ || $ordinal == 10 ]] || exit 2
mapfile -t counts < "${TRIGGER_COUNT_FILE:?set original verified parent counts}"
[[ ${#counts[@]} == 10 ]] || exit 2
export TRIGGER_EXPECTED_EVENTS=${counts[$((ordinal - 1))]}
export TRIGGER_SOURCE="$TRIGGER_ROOT/out/ce_regen_full/unfiltered_$ordinal.art"
export TRIGGER_TABLES="$TRIGGER_SPLIT/parent_$ordinal.root"
export TRIGGER_OUTPUT="${TRIGGER_TRANSPORT_OUTPUT_ROOT:-$TRIGGER_ROOT/out/ce_transport_full_v1}/parent_$ordinal"
unset TRIGGER_EXTRA_TABLES TRIGGER_SOURCE_MANIFEST
exec bash "$TRIGGER_CODE/trigger/ce_transport.sbatch"
