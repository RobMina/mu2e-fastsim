#!/bin/bash
# Combine every fixed shard and retain literal gate failures; no physics runs here.
set -euo pipefail
: "${CRV_TIER1B_ROOT:?}" "${CRV_TIER1B_CODE:?}"
R=$CRV_TIER1B_ROOT
C=$CRV_TIER1B_CODE
module load apptainer
A() {
 apptainer exec --cleanenv --env "PYTHONPATH=$C/core:$C/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1" \
 -B "$R" "${CRV_TIER1B_IMAGE:?project apptainer image}" python3 -m "$@"
}
case ${1:?hard|transport|chain|tests} in
hard)
 A valhist.device_tier1b_report "$R/default_ten_v18/summary.json" "$R/hard_bank_v20/summary.json" --out "$R/hard_full_v21.json"
 A valhist.device_tier1b_buffers_report "$R/default_ten_v18/summary.json" --out "$R/hard_warm_v21.json"
 ;;
transport)
 A valhist.device_transport_bank_report --summaries "$R/range_ten_v18/summary.json" "$R/range_bank_v20/summary.json" --out "$R/range_full_v21.json"
 ;;
chain)
 A valhist.device_tier1b_chain_report --numpy "$R/chain_default_v20_numpy" --jax "$R/chain_default_v20_jax" --out "$R/chain_default_medians_v21.json"
 A valhist.device_chain_compare --numpy "$R/chain_default_v20_numpy" --jax "$R/chain_default_v20_jax" --out "$R/chain_default_compare_v21.json"
 ;;
tests)
 A valhist.device_tier1b_chain_report --tests "$R/selected_v19.22195070.log" --out "$R/default_tests_v21.json"
 ;;
esac
