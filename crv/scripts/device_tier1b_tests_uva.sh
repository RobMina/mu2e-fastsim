#!/bin/bash
# CPU tests run in the existing image inside a Slurm allocation.
set -euo pipefail
: "${CRV_TIER1B_ROOT:?}" "${CRV_TIER1B_CODE:?}"
R=$CRV_TIER1B_ROOT
C=$CRV_TIER1B_CODE
H=${CRV_TIER1B_INPUTS:?input root with offline/, tier1b_strict_v2/ and the saved transport tables}
module load apptainer
cd "$C/crv"
apptainer exec --cleanenv \
 --env "PYTHONPATH=$C/core:$C/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_PLATFORMS=cpu,MU2E_SEARCH_PATH=$H/offline:$R/offline_cache/v13_39_00" \
 -B "$R" -B "$H" "${CRV_TIER1B_IMAGE:?project apptainer image}" \
 python3 -m pytest -q -p no:cacheprovider "$@"
