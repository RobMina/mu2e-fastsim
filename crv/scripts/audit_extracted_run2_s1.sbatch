#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Audit the complete no-transport extraction against its original log and catalog metadata.
set -euo pipefail
: "${FASTSIMKIT_SCRATCH:?set FASTSIMKIT_SCRATCH to the cluster working area (bound into the image)}"
: "${REUSE_CODE:?}" "${REUSE_CORE:?}" "${REUSE_ROOT:?}" "${PRODUCTION_TABLE:?}"
: "${PRODUCTION_METADATA:?}" "${PRODUCTION_LOG:?}" "${FASTSIM_IMAGE:?}" "${REUSE_GEOMETRY:?}"
module load apptainer
OUT="$REUSE_ROOT/production_audit_${SLURM_JOB_ID:?}.json"
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT"; exit 2; }
export APPTAINERENV_PYTHONPATH="$REUSE_CORE:$REUSE_CODE"
export APPTAINERENV_MU2E_SEARCH_PATH="$REUSE_GEOMETRY"
export APPTAINERENV_JAX_PLATFORMS=cpu APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_CRVFS_NO_FETCH=1
cd "$REUSE_CODE"
apptainer exec --cleanenv -B "$FASTSIMKIT_SCRATCH" "$FASTSIM_IMAGE" \
  python3 -m pytest -q -p no:cacheprovider tests/test_production_reuse.py tests/test_scatter_convergence.py \
  tests/test_run2_grid_plan.py tests/test_run2_grid_worker.py
apptainer exec --cleanenv -B "$FASTSIMKIT_SCRATCH" "$FASTSIM_IMAGE" \
  python3 -m crvfastsim.analyses.production_reuse \
  --production-table "$PRODUCTION_TABLE" --metadata "$PRODUCTION_METADATA" \
  --log "$PRODUCTION_LOG" --output "$OUT"
