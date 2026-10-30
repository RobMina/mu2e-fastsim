#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Audit saved reference precision and validate no-transport S1 FCL; no data discovery or grid submission.
set -euo pipefail
: "${REUSE_CODE:?}" "${REUSE_CORE:?}" "${REUSE_ROOT:?}" "${REFERENCE_ROOT:?}"
module load apptainer
OUT="$REUSE_ROOT/audit_${SLURM_JOB_ID:?}"
mkdir "$OUT"
cd "$REUSE_CODE"
E="PYTHONPATH=$REUSE_CORE:$REUSE_CODE,MU2E_SEARCH_PATH=${REUSE_GEOMETRY:?},JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,CRVFS_NO_FETCH=1,MPLCONFIGDIR=$OUT/mpl"
run() { apptainer exec --cleanenv --env "$E" -B "${REUSE_BIND_ROOT:?}" "${FASTSIM_IMAGE:?}" python3 "$@"; }
run -m pytest -q -p no:cacheprovider tests/test_production_reuse.py tests/test_scatter_convergence.py
run -m crvfastsim.analyses.production_reuse --reference-root "$REFERENCE_ROOT" \
  --metadata "$REUSE_ROOT/source517.metadata.json" --log "$REUSE_ROOT/production517.log" \
  --reference-worker-log "$REFERENCE_ROOT/thread_a_grid_continuation_20261025/pilot/tables/run2/worker.001400_00000102.log" \
  --output "$OUT/precision.json"
cd "$OUT"
export OFFLINE_EVENTS=0 OFFLINE_OUTPUT_FILE="$OUT/config_smoke.root" OFFLINE_LOG_PREFIX=config_smoke
bash "${OFFLINE_RUNNER:?}" "${OFFLINE_AREA:?}" "$REUSE_CODE/art/CrvFastSim/fcl/dump_run2_s1.fcl" "${OFFLINE_SMOKE_INPUT:?}"
