#!/bin/bash
# One-A100 warm CUDA profile; code snapshot is immutable for the job lifetime.
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:20:00
set -euo pipefail
: "${CRV_DEVICE_ROOT:?}" "${CRV_DEVICE_INPUTS:?}" "${FASTSIM_IMAGE:?}" "${CRV_DEVICE_CODE:?}" "${CRV_DEVICE_LABEL:?}"
R=$CRV_DEVICE_ROOT
H=$CRV_DEVICE_INPUTS
module load apptainer
nvidia-smi
NSYS_ROOT=$(dirname "$(dirname "$(readlink -f "$(command -v nsys)")")")
apptainer exec --nv --cleanenv \
  --env "PYTHONPATH=$CRV_DEVICE_CODE/core:$CRV_DEVICE_CODE/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,MU2E_SEARCH_PATH=$H/offline:$R/offline_cache/v13_39_00,CRVFS_DATA=$H/data,CRVFS_OFFLINE_CACHE=$R/offline_cache" \
  -B "$R" -B "$H" -B "$NSYS_ROOT" "$FASTSIM_IMAGE" \
  "$NSYS_ROOT/target-linux-x64/nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$R/${CRV_DEVICE_LABEL}_cuda" \
  python3 -m valhist.device_transport_profile \
  --gen "$H/gen_even/gen.001400_00000022.root" \
  --gdml "$H/data/geometry/mu2e_extracted_v04.gdml" \
  --counters "$H/offline/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt" \
  --repeat 5 --batch-size "${CRV_DEVICE_BATCH:-4096}" --cuda-capture --out "$R/$CRV_DEVICE_LABEL"
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  "$R/${CRV_DEVICE_LABEL}_cuda.nsys-rep" > "$R/${CRV_DEVICE_LABEL}_cuda_stats.txt"
