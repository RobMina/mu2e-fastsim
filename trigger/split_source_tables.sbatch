#!/bin/bash
# Partition saved source truth only; never reread art with a new simulation.
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_SOURCE_LIST:?}"
: "${TRIGGER_TABLES:?}"
: "${TRIGGER_EXTRA_TABLES:?}"
: "${TRIGGER_SPLIT:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
module load apptainer
if [[ -n ${TRIGGER_ANALYSIS_SHA256:-} ]]; then
  printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
fi
mapfile -t sources < "$TRIGGER_SOURCE_LIST"
mkdir -p "$(dirname "$TRIGGER_SPLIT")"
binds="$TRIGGER_CODE,$(dirname "$TRIGGER_SPLIT"),$(dirname "${sources[0]}"),$(dirname "$TRIGGER_TABLES"),$(dirname "$TRIGGER_EXTRA_TABLES")"
if [[ -n ${TRIGGER_RUNTIME_RECEIPT:-} ]]; then
  : "${TRIGGER_CORE:?}" "${TRIGGER_TRACKER:?}" "${TRIGGER_BEAM:?}"
  binds+=",$TRIGGER_CORE,$TRIGGER_TRACKER,$TRIGGER_BEAM,$(dirname "$TRIGGER_RUNTIME_RECEIPT")"
  export APPTAINERENV_PYTHONPATH="$TRIGGER_CORE:$TRIGGER_TRACKER:$TRIGGER_BEAM"
  export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_JAX_PLATFORMS=cpu
  apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON" "$TRIGGER_CODE/trigger/analysis_runtime.py" --output "$TRIGGER_RUNTIME_RECEIPT" --modules trkfastsim.validation.fieldce trkfastsim.validation.cegeometry trkfastsim.sim.fieldsteps beamfastsim.cechain mu2efastsim.tables
fi
apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON" "$TRIGGER_CODE/trigger/split_source_tables.py" --sources "${sources[@]}" --tables "$TRIGGER_TABLES" "$TRIGGER_EXTRA_TABLES" --output "$TRIGGER_SPLIT"
