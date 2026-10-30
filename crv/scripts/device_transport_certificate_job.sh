#!/bin/bash
# Audit retained transport tables without changing any producer or physics comparator.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 00:10:00
set -euo pipefail
: "${CRV_DEVICE_ROOT:?}" "${CRV_DEVICE_INPUTS:?}" "${CRV_DEVICE_CODE:?}" "${FASTSIM_IMAGE:?}"
R=$CRV_DEVICE_ROOT
H=$CRV_DEVICE_INPUTS
module load apptainer
apptainer exec --cleanenv \
  --env "PYTHONPATH=${CRV_DEVICE_REPORT_CODE:-$CRV_DEVICE_CODE/crv}:$CRV_DEVICE_CODE/core:$CRV_DEVICE_CODE/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1" \
  -B "$R" -B "$H" "$FASTSIM_IMAGE" \
  python3 -m valhist.device_transport_certificate \
  --summaries ${CRV_DEVICE_SUMMARIES:?set space-separated summary paths} \
  --inputs "$H/gen_even" --reference "$H/transport_v2" --stock "$H/deck/g4all" \
  --counters "$H/offline/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt" \
  --out "$R/${CRV_DEVICE_LABEL:?}_certificate.json"
