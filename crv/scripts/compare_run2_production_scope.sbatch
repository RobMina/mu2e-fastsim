#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Compare unmatched populations descriptively using existing tables; never run transport.
set -euo pipefail
: "${FASTSIMKIT_SCRATCH:?set FASTSIMKIT_SCRATCH to the cluster working area (bound into the image)}"
: "${REUSE_CODE:?}" "${REUSE_CORE:?}" "${REUSE_ROOT:?}" "${REUSE_GEOMETRY:?}"
: "${FASTSIM_IMAGE:?}" "${PRODUCTION_TABLE:?}" "${PRODUCTION_AUDIT:?}" "${REFERENCE_ROOT:?}" "${STAGE_LOG:?}"
module load apptainer
export APPTAINERENV_PYTHONPATH="$REUSE_CORE:$REUSE_CODE"
export APPTAINERENV_MU2E_SEARCH_PATH="$REUSE_GEOMETRY"
export APPTAINERENV_JAX_PLATFORMS=cpu APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_CRVFS_NO_FETCH=1
run() { apptainer exec --cleanenv -B "$FASTSIMKIT_SCRATCH" "$FASTSIM_IMAGE" python3 "$@"; }
cd "$REUSE_CODE"
run -m pytest -q -p no:cacheprovider tests/test_production_scope.py tests/test_production_reuse.py \
  tests/test_scatter_convergence.py tests/test_run2_grid_plan.py tests/test_run2_grid_worker.py
run -m crvfastsim.analyses.production_scope --production-table "$PRODUCTION_TABLE" \
  --production-audit "$PRODUCTION_AUDIT" --reference-root "$REFERENCE_ROOT" --stage-log "$STAGE_LOG" \
  --output "$REUSE_ROOT/population_scope_${SLURM_JOB_ID:?}.json"
