#!/bin/bash
set -eo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
O=/exp/mu2e/data/users/ram2aq/crv_e2e_gpu13_20261027
D=/exp/mu2e/data/users/ram2aq/crv_fastsim
PY=$D/venv_eaf/bin/python3
export PYTHONPATH=$(dirname "$HERE"):$(dirname "$(dirname "$HERE")")/core_e2e_gpu13
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CRVFASTSIM_COMPACT_DISPATCH=1
"$PY" -m crvfastsim reco-vr3 "$D/valhist/g4ref_v1/a0_dump/valhist.001400_00000011.root" \
    --both --outdir "$O/eaf_vr3_stock_s11"
for compact in 0 1; do
    bash "$HERE/agent13_local_chain.sh" gpu "$O/eaf_50k_compact$compact" 50000 "$compact"
done
"$PY" "$HERE/agent13_compare_bench.py" "$O/eaf_50k_compact0" "$O/eaf_50k_compact1" \
    --allow-derived-rounding --out "$O/eaf_regression_50k.json"
