#!/bin/bash
# Fixed batch-size sweep for warm transport, not physics seed selection.
# Slurm account: set SBATCH_ACCOUNT (see site.env.example) or pass sbatch -A.
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
set -euo pipefail
: "${CRV_DEVICE_ROOT:?}" "${CRV_DEVICE_INPUTS:?}" "${FASTSIM_IMAGE:?}" "${CRV_DEVICE_CODE:?}"
R=$CRV_DEVICE_ROOT
H=$CRV_DEVICE_INPUTS
module load apptainer
for batch in 2048 4096 8192; do
  apptainer exec --nv --cleanenv \
    --env "PYTHONPATH=$CRV_DEVICE_CODE/core:$CRV_DEVICE_CODE/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1" \
    -B "$R" -B "$H" "$FASTSIM_IMAGE" \
    python3 -m valhist.device_transport_profile --gen "$H/gen_even/gen.001400_00000022.root" \
    --gdml "$H/data/geometry/mu2e_extracted_v04.gdml" \
    --counters "$H/offline/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt" \
    --repeat 5 --batch-size "$batch" --out "$R/batch_probe_$batch"
done
