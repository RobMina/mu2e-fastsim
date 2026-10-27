#!/bin/bash
set -eo pipefail
umask 077
MODE=${1:?cpu or gpu}
OUT=${2:?new absolute output directory}
N=${3:-5000}
COMPACT=${4:-0}
D=/exp/mu2e/data/users/ram2aq/crv_fastsim
CODE=$(cd "$(dirname "$0")/.." && pwd)
CORE=$(dirname "$CODE")/core_e2e_gpu13
PY=$D/venv_eaf/bin/python3
export PYTHONPATH=$CODE:$CORE
export CRVFS_LOOKUP_DIR=$D/lookup_v6_0 CRVFS_DATA=$D
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1
export CRVFASTSIM_COMPACT_DISPATCH=$COMPACT
args=()
if [ "${AGENT13_REUSE_CONTEXT:-0}" = 1 ]; then args+=(--reuse-context); fi
if [ "$MODE" = cpu ]; then
    export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
    exec taskset -c 0 bash "$0" cpu-pinned "$OUT" "$N" "$COMPACT"
elif [ "$MODE" = gpu ]; then
    args+=(--jax)
elif [ "$MODE" != cpu-pinned ]; then
    echo 'Unknown device mode' >&2
    exit 2
fi
"$PY" -m crvfastsim.analyses.e2e_bench "$D/g4ref_v1_tables/gen.001400_00000011.root" \
    --library "$D/tier1b_v7/lib_prod_as_20files.root" \
    --calib "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv" \
    --deviation "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" \
    --gdml "$D/geometry/mu2e_extracted_v04.gdml" \
    --run-config extracted_v04 --window offspill --seed 20261027 \
    --max-events "$N" --repeat "${AGENT13_REPEAT:-2}" --outdir "$OUT" "${args[@]}"
