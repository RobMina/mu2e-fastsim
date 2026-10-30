#!/bin/bash
# New-default validation in the existing project image, on an allocated UVA node.
set -euo pipefail
: "${CRV_TIER1B_ROOT:?}" "${CRV_TIER1B_CODE:?}" "${CRV_TIER1B_LABEL:?}"
R=$CRV_TIER1B_ROOT
C=$CRV_TIER1B_CODE
H=${CRV_TIER1B_INPUTS:?input root with offline/, tier1b_strict_v2/ and the saved transport tables}
module load apptainer
cd "$C/crv"
apptainer exec --nv --cleanenv \
 --env "PYTHONPATH=$C/core:$C/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_PLATFORMS=cuda,MU2E_SEARCH_PATH=$H/offline" \
 -B "$R" -B "$H" "${CRV_TIER1B_IMAGE:?project apptainer image}" \
 python3 -m valhist.device_transport --inputs "$H/gen_even" --reference "$H/transport_tdsr4" \
 --gdml "$H/data/geometry/mu2e_extracted_v04.gdml" \
 --counters "$H/offline/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt" \
 --stock "$H/deck/g4all" --range-out --reuse-baseline --no-candidate-tables \
 --out "$R/$CRV_TIER1B_LABEL" "$@"
