#!/bin/bash
set -eo pipefail
umask 077
O=/exp/mu2e/data/users/ram2aq/crv_e2e_gpu13_20261027
mkdir -p "$O"
PY=/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf/bin/python3
export PYTHONPATH=/exp/mu2e/app/users/ram2aq/analysis/crv_wt/e2e_gpu13:/exp/mu2e/app/users/ram2aq/analysis/crv_wt/core_e2e_gpu13
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
hostname
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PY" -c 'import jax, crvfastsim, mu2efastsim; print(jax.__version__, jax.devices(), crvfastsim.__file__, mu2efastsim.__file__); print(jax.devices()[0].memory_stats()); assert jax.default_backend() == "gpu"'
D=/exp/mu2e/data/users/ram2aq/crv_fastsim
for f in "$D/g4ref_v1_tables/gen.001400_00000011.root" "$D/tier1b_v7/lib_prod_as_20files.root" "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv" "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" "$D/geometry/mu2e_extracted_v04.gdml"; do
    test -r "$f" || exit 3
    stat -c '%s %n' "$f"
done
