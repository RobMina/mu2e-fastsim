#!/bin/bash
# Fetch optics once into job-local memory, benchmark both transport paths, then remove them.
set -euo pipefail
: "${CRV_DEVICE_ROOT:?}" "${CRV_DEVICE_INPUTS:?}" "${FASTSIM_IMAGE:?}"
CODE=${CRV_DEVICE_CODE:-$CRV_DEVICE_ROOT/code}
module load apptainer
L=$(mktemp -d -t crv_device_transport.XXXXXX)
trap 'rm -rf -- "$L"' EXIT
export CRV_DEVICE_OPTICS=$L CRV_DEVICE_OPTICS_BIND="-B $L" CRV_DEVICE_NV=--nv
F() {
    apptainer exec --cleanenv --env "PYTHONPATH=$CODE/core" -B "$CRV_DEVICE_ROOT" -B "$L" "$FASTSIM_IMAGE" python3 -m mu2efastsim.cvmfsget "$@"
}
for file in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
    F mu2e.opensciencegrid.org "/DataFiles/CRVConditions/v6_0/$file" "$L/$file"
done
mkdir -p "$CRV_DEVICE_ROOT/offline_cache/v13_39_00/Offline/CRVResponse/data"
F mu2e.opensciencegrid.org /Musings/Offline/v13_39_00/Offline/CRVResponse/data/singlePEWaveform_v3.txt \
  "$CRV_DEVICE_ROOT/offline_cache/v13_39_00/Offline/CRVResponse/data/singlePEWaveform_v3.txt"
sha256sum "$L"/* > "$CRV_DEVICE_ROOT/${CRV_DEVICE_LABEL:?}_optics.sha256"
bash "$CODE/crv/scripts/device_transport_uva.sh" chain
