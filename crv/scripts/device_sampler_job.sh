#!/bin/bash
# Paired GPU unit tests and full-size fluctuation-law validation in the project image.
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
set -euo pipefail
: "${CRV_DEVICE_ROOT:?}" "${CRV_DEVICE_CODE:?}" "${FASTSIM_IMAGE:?}" "${CRV_DEVICE_LABEL:?}"
module load apptainer
A() {
  apptainer exec --nv --cleanenv \
    --env "PYTHONPATH=$CRV_DEVICE_CODE/core:$CRV_DEVICE_CODE/crv,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1" \
    -B "$CRV_DEVICE_ROOT" "$FASTSIM_IMAGE" "$@"
}
cd "$CRV_DEVICE_CODE/crv"
A python3 -m pytest tests/test_eloss_transport_jax.py tests/test_crossings_device_order.py tests/test_transport_jax.py -q -p no:cacheprovider
A python3 -m valhist.device_sampler_validation --out "$CRV_DEVICE_ROOT/${CRV_DEVICE_LABEL}_sampler.json"
