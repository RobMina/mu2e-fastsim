#!/bin/bash
# Project-image CPU tests or one-A100 validation/chain timing inside a Slurm allocation.
set -euo pipefail
: "${CRV_DEVICE_ROOT:?set output/code snapshot root}" "${CRV_DEVICE_INPUTS:?set existing shared input root}" "${FASTSIM_IMAGE:?set image}"
R=$CRV_DEVICE_ROOT
H=$CRV_DEVICE_INPUTS
CODE=${CRV_DEVICE_CODE:-$R/code}
module load apptainer
A() {
    apptainer exec ${CRV_DEVICE_NV:-} --cleanenv \
      --env "PYTHONPATH=$CODE/core:$CODE/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,MU2E_SEARCH_PATH=$H/offline:$R/offline_cache/v13_39_00,CRVFS_DATA=$H/data,CRVFS_LOOKUP_DIR=${CRV_DEVICE_OPTICS:-$H/data/lookup},CRVFS_OFFLINE_CACHE=$R/offline_cache,JAX_PLATFORMS=${JAX_PLATFORMS:-}" \
      -B "$R" -B "$H" ${CRV_DEVICE_OPTICS_BIND:-} "$FASTSIM_IMAGE" "$@"
}
cd "$CODE/crv"
case ${1:?unit|closure|chain|selected} in
unit) A python3 -m pytest tests/test_transport_jax.py -q -p no:cacheprovider ;;
selected)
    test -s "$R/selected_tests.txt"
    failed=0
    while read -r test; do
        echo "== $test"
        A python3 -m pytest "$test" -q -p no:cacheprovider || failed=1
    done < "$R/selected_tests.txt"
    exit "$failed" ;;
closure) A python3 -m valhist.device_transport --inputs "$H/gen_even" --reference "$H/transport_v2" \
    --gdml "$H/data/geometry/mu2e_extracted_v04.gdml" \
    --counters "$H/offline/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt" \
    --stock "$H/deck/g4all" --shards ${CRV_DEVICE_SHARDS:-22 24 26} --repeat 1 --out "$R/${CRV_DEVICE_LABEL:?}" ;;
chain)
    : "${CRV_DEVICE_OPTICS:?set node-local optics directory}"
    inputs=()
    for shard in ${CRV_DEVICE_CHAIN_SHARDS:-22}; do
        printf -v suffix '%08d' "$shard"
        inputs+=("$H/gen_even/gen.001400_$suffix.root")
    done
    for backend in ${CRV_DEVICE_BACKENDS:-numpy jax}; do
        A python3 -m crvfastsim e2e-bench "${inputs[@]}" \
          --calib "$H/data/geometry/crvcalib_sim_best_v1_5_run1400.csv" \
          --deviation "$H/data/geometry/crvphoton_sim_best_v1_5_run1400.csv" \
          --gdml "$H/data/geometry/mu2e_extracted_v04.gdml" \
          --tier1b strict_v3 --tier1b-dir "$H/tier1b_strict_v2" --jax \
          --transport-backend "$backend" ${CRV_DEVICE_PREFETCH:+--prefetch} --reuse-context --repeat "${CRV_DEVICE_REPEAT:-2}" --max-events "${CRV_DEVICE_CHAIN_EVENTS:-1000}" \
          --outdir "$R/${CRV_DEVICE_LABEL:?}_$backend"
    done ;;
*) exit 2 ;;
esac
