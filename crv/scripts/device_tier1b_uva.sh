#!/bin/bash
# Run only inside a UVA Slurm allocation with the existing project image.
set -euo pipefail
: "${CRV_TIER1B_ROOT:?}" "${CRV_TIER1B_CODE:?}" "${CRV_TIER1B_LABEL:?}"
R=$CRV_TIER1B_ROOT
C=$CRV_TIER1B_CODE
H=${CRV_TIER1B_INPUTS:?input root with offline/, tier1b_strict_v2/ and the saved transport tables}
module load apptainer
cd "$C/crv"
apptainer exec ${CRV_TIER1B_NV:-} --cleanenv \
  --env "PYTHONPATH=$C/core:$C/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_PLATFORMS=${JAX_PLATFORMS:-cpu},MU2E_SEARCH_PATH=$H/offline" \
  -B "$R" -B "$H" "${CRV_TIER1B_IMAGE:?project apptainer image}" \
  python3 -m valhist.device_tier1b --counters "$H/offline/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt" \
  --library-dir "$H/tier1b_strict_v2" --transport "${CRV_TIER1B_TRANSPORT:-$H/transport_tdsr4}" --out "$R/$CRV_TIER1B_LABEL" "$@"
