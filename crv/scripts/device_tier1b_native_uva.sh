#!/bin/bash
# GPU stage-chain smoke; this still has host table boundaries.
set -euo pipefail
: "${CRV_TIER1B_ROOT:?}" "${CRV_TIER1B_CODE:?}" "${CRV_TIER1B_LABEL:?}"
R=$CRV_TIER1B_ROOT
C=$CRV_TIER1B_CODE
H=${CRV_TIER1B_INPUTS:?input root with offline/, tier1b_strict_v2/ and the saved transport tables}
I=${CRV_TIER1B_CRY_IMAGE:?image with the native CRY overlay}
module load apptainer
L=$(mktemp -d -t crv_device_native.XXXXXX)
trap 'rm -rf -- "$L"' EXIT
F() {
 apptainer exec --cleanenv --env "PYTHONPATH=$C/core" -B "$R" -B "$L" "$I" python3 -m mu2efastsim.cvmfsget "$@"
}
for file in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
 F mu2e.opensciencegrid.org "/DataFiles/CRVConditions/v6_0/$file" "$L/$file"
done
apptainer exec --nv --cleanenv \
 --env "PYTHONPATH=$C/core:$C/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,MU2E_SEARCH_PATH=$H/offline:$R/offline_cache/v13_39_00,CRVFS_DATA=$H/data,CRVFS_LOOKUP_DIR=$L,CRVFS_OFFLINE_CACHE=$R/offline_cache,JAX_PLATFORMS=cuda" \
 -B "$R" -B "$H" -B "$L" "$I" \
 python3 -m valhist.device_tier1b_native \
 --calib "$H/data/geometry/crvcalib_sim_best_v1_5_run1400.csv" \
 --deviation "$H/data/geometry/crvphoton_sim_best_v1_5_run1400.csv" \
 --gdml "$H/data/geometry/mu2e_extracted_v04.gdml" --library-dir "$H/tier1b_strict_v2" \
 --events "${CRV_TIER1B_NATIVE_EVENTS:-10000}" --repeat 3 --out "$R/${CRV_TIER1B_LABEL}.json"
