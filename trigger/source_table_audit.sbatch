#!/bin/bash
# Audit all exported source births and represented straw geometry without rerunning art.
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_SOURCE_LIST:?}"
: "${TRIGGER_TABLES:?}"
: "${TRIGGER_EXTRA_TABLES:?}"
: "${TRIGGER_TRACKER:?}"
: "${TRIGGER_CORE:?}"
: "${TRIGGER_GDML:?}"
: "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_EXPECTED_EVENTS:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
[[ ! -e $TRIGGER_OUTPUT ]] || exit 2
module load apptainer
mkdir "$TRIGGER_OUTPUT"
mapfile -t sources < "$TRIGGER_SOURCE_LIST"
export APPTAINERENV_PYTHONPATH="$TRIGGER_CORE:$TRIGGER_TRACKER"
export APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_OMP_NUM_THREADS=1
binds="$TRIGGER_CODE,$TRIGGER_OUTPUT,$TRIGGER_TRACKER,$TRIGGER_CORE,$(dirname "$TRIGGER_SOURCE_LIST"),$(dirname "${sources[0]}"),$(dirname "$TRIGGER_TABLES"),$(dirname "$TRIGGER_EXTRA_TABLES"),$(dirname "$TRIGGER_GDML")"
apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON" "$TRIGGER_CODE/trigger/source_table_audit.py" --sources "${sources[@]}" --tables "$TRIGGER_TABLES" "$TRIGGER_EXTRA_TABLES" --gdml "$TRIGGER_GDML" --expected-events "$TRIGGER_EXPECTED_EVENTS" --origin -3904 0 10171 --output "$TRIGGER_OUTPUT/source-table-audit.json"
