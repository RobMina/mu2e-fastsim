#!/bin/bash
# Read private inputs in place and retain new validation products only in the owned FNAL data area.
set -eo pipefail
umask 077
O=${1:?new absolute output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
D=/exp/mu2e/data/users/ram2aq/crv_fastsim
BASE=/exp/mu2e/data/users/ram2aq/crv_e2e_gpu13_20261027
PY=$D/venv_eaf/bin/python3
case "$O" in "$BASE"/*) ;; *) echo 'Output must be in agent13 data area'; exit 2;; esac
mkdir "$O"
mkdir "$O/summary" "$O/references"
for compact in 0 1; do
    cp "$BASE/eaf_50k_compact$compact/benchmark.json" "$O/references/n50000_compact$compact.json"
done
export PYTHONPATH=$(dirname "$HERE"):$(dirname "$(dirname "$HERE")")/core_e2e_gpu13
export CRVFS_DATA=$D CRVFS_LOOKUP_DIR=$D/lookup_v6_0
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1
hostname > "$O/summary/host.txt"
nvidia-smi -L > "$O/summary/gpu.txt"
"$PY" -c 'import jax; print(jax.__version__, jax.devices()); print(jax.devices()[0].memory_stats()); assert jax.default_backend() == "gpu"' > "$O/summary/device.txt"
"$PY" "$HERE/agent13_context_campaign.py" --data "$D" --lookup-dir "$D/lookup_v6_0" \
    --inputs-dir "$D/g4ref_v1_tables" --library "$D/tier1b_v7/lib_prod_as_20files.root" \
    --calib "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv" \
    --deviation "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" \
    --gdml "$D/geometry/mu2e_extracted_v04.gdml" --outdir "$O/details" \
    --summary-dir "$O/summary" --reference-dir "$O/references"
