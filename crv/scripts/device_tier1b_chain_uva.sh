#!/bin/bash
# Audited cached-primary chain with selectable device transport and tier1b.
set -euo pipefail
: "${CRV_TIER1B_ROOT:?}" "${CRV_TIER1B_CODE:?}" "${CRV_TIER1B_LABEL:?}"
R=$CRV_TIER1B_ROOT
C=$CRV_TIER1B_CODE
H=${CRV_TIER1B_INPUTS:?input root with offline/, tier1b_strict_v2/ and the saved transport tables}
module load apptainer
L=$(mktemp -d -t crv_device_tier1b.XXXXXX)
trap 'rm -rf -- "$L"' EXIT
F() {
 apptainer exec --cleanenv --env "PYTHONPATH=$C/core" -B "$R" -B "$L" \
 "${CRV_TIER1B_IMAGE:?project apptainer image}" python3 -m mu2efastsim.cvmfsget "$@"
}
for file in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
 F mu2e.opensciencegrid.org "/DataFiles/CRVConditions/v6_0/$file" "$L/$file"
done
mkdir -p "$R/offline_cache/v13_39_00/Offline/CRVResponse/data"
wave="$R/offline_cache/v13_39_00/Offline/CRVResponse/data/singlePEWaveform_v3.txt"
if test ! -s "$wave"; then
 F mu2e.opensciencegrid.org /Musings/Offline/v13_39_00/Offline/CRVResponse/data/singlePEWaveform_v3.txt "$wave"
fi
sha256sum "$L"/* > "$R/${CRV_TIER1B_LABEL}_optics.sha256"
inputs=()
for shard in ${CRV_TIER1B_CHAIN_SHARDS:-22}; do
 printf -v suffix '%08d' "$shard"
 inputs+=("$H/gen_even/gen.001400_$suffix.root")
done
for backend in ${CRV_TIER1B_BACKENDS:-numpy jax}; do
 transport=jax
 if test "$backend" = numpy; then transport=${CRV_TIER1B_NUMPY_TRANSPORT:-jax}; fi
 apptainer exec --nv --cleanenv \
 --env "PYTHONPATH=$C/core:$C/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,MU2E_SEARCH_PATH=$H/offline:$R/offline_cache/v13_39_00,CRVFS_DATA=$H/data,CRVFS_LOOKUP_DIR=$L,CRVFS_OFFLINE_CACHE=$R/offline_cache,JAX_PLATFORMS=cuda" \
 -B "$R" -B "$H" -B "$L" "${CRV_TIER1B_IMAGE:?project apptainer image}" \
 python3 -m crvfastsim e2e-bench "${inputs[@]}" \
 --calib "$H/data/geometry/crvcalib_sim_best_v1_5_run1400.csv" \
 --deviation "$H/data/geometry/crvphoton_sim_best_v1_5_run1400.csv" \
 --gdml "$H/data/geometry/mu2e_extracted_v04.gdml" \
 --tier1b "${CRV_TIER1B_SET:-strict_v4}" --tier1b-dir "$H/tier1b_strict_v2" --jax \
 --transport-backend "$transport" --tier1b-backend "$backend" --reuse-context \
 --repeat "${CRV_TIER1B_REPEAT:-4}" --max-events "${CRV_TIER1B_CHAIN_EVENTS:-50000}" \
 --outdir "$R/${CRV_TIER1B_LABEL}_$backend"
done
