#!/bin/bash
# Run the explicit-timing tracker comparison in an existing project image.
set -eo pipefail
: "${MIXING_CODE:?umbrella snapshot}"
: "${TRACKER_CODE:?tracker snapshot}"
: "${CORE_CODE:?core snapshot}"
: "${MIXING_SIF:?project image}"
: "${MIXING_DATA_ROOT:?canonical shared input/output root}"
module load apptainer
args=(exec --cleanenv)
if [[ ${MIXING_GPU:-0} == 1 ]]; then args+=(--nv); fi
args+=(-B "$MIXING_DATA_ROOT:$MIXING_DATA_ROOT" --env "PYTHONPATH=$MIXING_CODE:$TRACKER_CODE:$CORE_CODE")
args+=(--env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env PYTHONDONTWRITEBYTECODE=1)
args+=(--env XLA_PYTHON_CLIENT_PREALLOCATE=false)
if [[ -n ${JAX_COMPILATION_CACHE_DIR:-} ]]; then
  args+=(--env "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR")
fi
apptainer "${args[@]}" "$MIXING_SIF" python3 -m mixing.fast_tracker "$@"
