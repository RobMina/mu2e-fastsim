#!/bin/bash
#SBATCH -J calo_shower_baseline
#SBATCH -t 00:30:00
set -eo pipefail
: "${CALO_SHOWER_CODE:?}" "${CALO_SHOWER_CORE:?}" "${CALO_SHOWER_INPUT:?}" "${CALO_SHOWER_OUT:?}" "${CALO_SHOWER_IMAGE:?}"
: "${CALO_SHOWER_DEVICE:?}" "${CALO_RECO_INPUT:?}" "${CALO_STOCK_NTUPLE:?}" "${CALO_STOCK_NTUPLE2:?}" "${CALO_DEVELOP_STEPS:?}"
module load apptainer
mkdir -p "$CALO_SHOWER_OUT"
export APPTAINERENV_PYTHONPATH="$CALO_SHOWER_CODE/calo:$CALO_SHOWER_CODE/beam:$CALO_SHOWER_CORE"
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_JAX_ENABLE_X64=true APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_JAX_PLATFORMS=cpu
export APPTAINERENV_XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0"
extra=()
if [ "$CALO_SHOWER_DEVICE" = gpu ]; then
  extra+=(--nv)
  export APPTAINERENV_JAX_PLATFORMS=cuda APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  nvidia-smi -L
fi
binds="$CALO_SHOWER_CODE,$CALO_SHOWER_CORE,$CALO_SHOWER_INPUT,$CALO_SHOWER_OUT,$CALO_RECO_INPUT,$CALO_STOCK_NTUPLE,$CALO_STOCK_NTUPLE2,$CALO_DEVELOP_STEPS"
/usr/bin/time -v -o "$CALO_SHOWER_OUT/process.time" apptainer exec --cleanenv "${extra[@]}" --bind "$binds" "$CALO_SHOWER_IMAGE" \
  python3 -m calofastsim.showers.develop_reco --steps "$CALO_DEVELOP_STEPS" \
  --inputs "$CALO_SHOWER_INPUT/chain_inputs.npz" --reco-code "$CALO_SHOWER_CODE/calo/calofastsim" \
  --reco-inputs "$CALO_RECO_INPUT" --stock-ntuple "$CALO_STOCK_NTUPLE" --stock-ntuple2 "$CALO_STOCK_NTUPLE2" \
  --out "$CALO_SHOWER_OUT" --device "$CALO_SHOWER_DEVICE" --repeat 3 --seed 20261108 --save-clusters
