#!/bin/bash
#SBATCH -J calo_shower_profile
#SBATCH -t 00:30:00
set -eo pipefail
: "${CALO_SHOWER_CODE:?}" "${CALO_SHOWER_CORE:?}" "${CALO_SHOWER_INPUT:?}" "${CALO_SHOWER_OUT:?}" "${CALO_SHOWER_IMAGE:?}"
: "${CALO_SHOWER_DEVICE:?}" "${CALO_RECO_INPUT:?}" "${CALO_STOCK_NTUPLE:?}" "${CALO_SHOWER_MAPS:?}" "${CALO_SHOWER_UMBRELLA_SHA:?}" "${CALO_SHOWER_CORE_SHA:?}"
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
binds="$CALO_SHOWER_CODE,$CALO_SHOWER_CORE,$CALO_SHOWER_INPUT,$CALO_SHOWER_OUT,$CALO_RECO_INPUT,$CALO_STOCK_NTUPLE,$CALO_SHOWER_MAPS"
run=(apptainer exec --cleanenv "${extra[@]}" --bind "$binds" "$CALO_SHOWER_IMAGE" python3)
"${run[@]}" -m calofastsim.showers.measure_chain --inputs "$CALO_SHOWER_INPUT/chain_inputs.npz" \
  --library "$CALO_SHOWER_INPUT/ce_showers_dev.npz" --geometry "$CALO_SHOWER_INPUT/geometry.txt" \
  --maps "$CALO_SHOWER_MAPS/DSMap.header" "$CALO_SHOWER_MAPS/TSdMap.header" \
  --response-source "$CALO_SHOWER_CODE/calo/calofastsim/response.py" --out "$CALO_SHOWER_OUT/transport" \
  --arm development --backend device --device "$CALO_SHOWER_DEVICE" --repeat 1 --seed 20261105 \
  --digitization-start 450 --digitization-end 1795 --digitization-buffer 10 \
  --umbrella-commit "$CALO_SHOWER_UMBRELLA_SHA" --core-commit "$CALO_SHOWER_CORE_SHA"
for placement in geometric crystal; do
  mkdir -p "$CALO_SHOWER_OUT/$placement"
  /usr/bin/time -v -o "$CALO_SHOWER_OUT/$placement/process.time" "${run[@]}" -m calofastsim.showers.reconstruct \
    --chain "$CALO_SHOWER_OUT/transport/chain.npz" --chain-report "$CALO_SHOWER_OUT/transport/numbers.json" \
    --inputs "$CALO_SHOWER_INPUT/chain_inputs.npz" --library "$CALO_SHOWER_INPUT/ce_showers_dev.npz" \
    --maps "$CALO_SHOWER_MAPS/DSMap.header" "$CALO_SHOWER_MAPS/TSdMap.header" \
    --reco-code "$CALO_SHOWER_CODE/calo/calofastsim" --reco-inputs "$CALO_RECO_INPUT" \
    --reco-commit "$CALO_SHOWER_UMBRELLA_SHA" --stock-ntuple "$CALO_STOCK_NTUPLE" \
    --live-transport device --transport-geometry "$CALO_SHOWER_INPUT/geometry.txt" \
    --placement "$placement" --out "$CALO_SHOWER_OUT/$placement" --device "$CALO_SHOWER_DEVICE" --repeat 3 --seed 20261105
done
