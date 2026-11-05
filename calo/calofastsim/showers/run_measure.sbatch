#!/bin/bash
#SBATCH -J calo_shower_measure
#SBATCH -t 01:00:00
set -eo pipefail
: "${CALO_SHOWER_CODE:?}" "${CALO_SHOWER_CORE:?}" "${CALO_SHOWER_INPUT:?}" "${CALO_SHOWER_OUT:?}" "${CALO_SHOWER_MAPS:?}" "${CALO_SHOWER_IMAGE:?}"
: "${CALO_SHOWER_BACKEND:?}" "${CALO_SHOWER_DEVICE:?}" "${CALO_SHOWER_UMBRELLA_SHA:?}" "${CALO_SHOWER_CORE_SHA:?}"
module load apptainer
mkdir -p "$CALO_SHOWER_OUT"
export APPTAINERENV_PYTHONPATH="$CALO_SHOWER_CODE/calo:$CALO_SHOWER_CODE/beam:$CALO_SHOWER_CORE"
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_JAX_ENABLE_X64=true APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_JAX_PLATFORMS=cpu
if [ "$CALO_SHOWER_DEVICE" = gpu ]; then export APPTAINERENV_JAX_PLATFORMS=cuda; fi
export APPTAINERENV_XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0"
extra=()
if [ "$CALO_SHOWER_DEVICE" = gpu ]; then
  extra+=(--nv)
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  nvidia-smi -L
fi
/usr/bin/time -v -o "$CALO_SHOWER_OUT/process.time" apptainer exec --cleanenv "${extra[@]}" \
  --bind "$CALO_SHOWER_CODE,$CALO_SHOWER_CORE,$CALO_SHOWER_INPUT,$CALO_SHOWER_OUT,$CALO_SHOWER_MAPS" "$CALO_SHOWER_IMAGE" \
  python3 -m calofastsim.showers.measure_chain \
  --inputs "$CALO_SHOWER_INPUT/chain_inputs.npz" --library "$CALO_SHOWER_INPUT/ce_showers_dev.npz" \
  --geometry "$CALO_SHOWER_INPUT/geometry.txt" --maps "$CALO_SHOWER_MAPS/DSMap.header" "$CALO_SHOWER_MAPS/TSdMap.header" \
  --response-source "$CALO_SHOWER_CODE/response.py" --out "$CALO_SHOWER_OUT" --arm heldout \
  --backend "$CALO_SHOWER_BACKEND" --device "$CALO_SHOWER_DEVICE" --repeat 3 --seed 20261105 \
  --digitization-start 450 --digitization-end 1795 --digitization-buffer 10 \
  --umbrella-commit "$CALO_SHOWER_UMBRELLA_SHA" --core-commit "$CALO_SHOWER_CORE_SHA"
