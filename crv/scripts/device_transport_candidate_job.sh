#!/bin/bash
# Frozen candidate tests or held-out physics and repeated synchronized timing.
set -euo pipefail
: "${CRV_DEVICE_ROOT:?}" "${CRV_DEVICE_INPUTS:?}" "${FASTSIM_IMAGE:?}" "${CRV_DEVICE_CODE:?}" "${CRV_DEVICE_LABEL:?}"
R=$CRV_DEVICE_ROOT
H=$CRV_DEVICE_INPUTS
module load apptainer
A() {
  apptainer exec ${CRV_DEVICE_NV:-} --cleanenv \
    --env "PYTHONPATH=$CRV_DEVICE_CODE/core:$CRV_DEVICE_CODE/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_PLATFORMS=${JAX_PLATFORMS:-},MU2E_SEARCH_PATH=$H/offline:$R/offline_cache/v13_39_00,CRVFS_DATA=$H/data,CRVFS_OFFLINE_CACHE=$R/offline_cache" \
    -B "$R" -B "$H" "$FASTSIM_IMAGE" "$@"
}
cd "$CRV_DEVICE_CODE/crv"
case ${1:?tests|closure} in
tests) A python3 -m pytest tests/test_eloss_transport_jax.py tests/test_crossings_device_order.py tests/test_transport_jax.py -q -p no:cacheprovider ;;
closure) A python3 -m valhist.device_transport --inputs "$H/gen_even" --reference "$H/transport_v2" \
    --gdml "$H/data/geometry/mu2e_extracted_v04.gdml" \
    --counters "$H/offline/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt" \
    --stock "$H/deck/g4all" --shards ${CRV_DEVICE_SHARDS:-22 24 26} \
    ${CRV_DEVICE_SUMMARY_ONLY:+--no-candidate-tables} \
    ${CRV_DEVICE_REUSE_BASELINE:+--reuse-baseline} \
    ${CRV_DEVICE_PAIRED_REFERENCE:+--paired-jax-reference "$CRV_DEVICE_PAIRED_REFERENCE"} \
    --repeat "${CRV_DEVICE_REPEAT:-5}" --numpy-repeat 5 --batch-size "${CRV_DEVICE_BATCH:-4096}" --out "$R/$CRV_DEVICE_LABEL" ;;
*) exit 2 ;;
esac
