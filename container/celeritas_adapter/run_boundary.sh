#!/bin/bash
#SBATCH --job-name=m2fs_n5_boundary
#SBATCH --account=ldmxuva
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/mu2efastsim/probe/n5-boundary-%j.out
# Run public thin-surface diagnostics with the existing adapter, never compile.
set -eo pipefail
module load apptainer
set -u
: "${SLURM_JOB_ID:?Submit with sbatch}"
INPUT=/home/$USER/mu2e/fastsim_uva/n5_boundary_20261027
SRC=/home/$USER/mu2e/fastsim_uva/celer_adapter_h4
H5=/scratch/$USER/mu2efastsim/celer_rdc_h5
IMAGE=/scratch/$USER/mu2efastsim/mu2e-celeritas.sif
BASE=/scratch/$USER/mu2efastsim/probe
OUT=$BASE/n5_boundary_${SLURM_JOB_ID}
GPU=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)
[[ $GPU == *A100* ]] || { echo 'A100 required' >&2; exit 2; }
ulimit -c 0
mkdir "$OUT"
printf 'job=%s host=%s utc=%s\n%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" "$GPU" | tee "$OUT/allocation.txt"
(cd "$INPUT" && sha256sum --check snapshot.sha256)
cp -r "$INPUT/container" "$OUT/replay-source"
cp "$INPUT/snapshot.sha256" "$OUT/"
srun --ntasks=1 apptainer exec --cleanenv --nv \
  --bind "$BASE:$BASE" --bind "$INPUT:$INPUT:ro" --bind "$SRC:$SRC:ro" --bind "$H5:$H5:ro" \
  "$IMAGE" bash -c 'export LD_LIBRARY_PATH="$1/lib64:$1/lib:${LD_LIBRARY_PATH:-}"; exec python3 "$2/container/celeritas_adapter/boundary.py" run "$3/results" --source "$4" --build "$5" --prefix "$1"' \
  bash "$H5/install" "$INPUT" "$OUT" "$SRC" "$H5/adapter-build"
