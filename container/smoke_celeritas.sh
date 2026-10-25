#!/bin/bash
#SBATCH --job-name=m2fs_celer_smoke
#SBATCH --account=ldmxuva
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/mu2efastsim/probe/celer-smoke-%j.out
# Run the public synthetic EM test in the existing image, without building or installing anything.
set -eo pipefail
module load apptainer
set -u
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
HERE=$(cd "$(dirname "$1")" && pwd)
SCRIPT=$HERE/$(basename "$1")
IMAGE=$(realpath /scratch/$USER/mu2efastsim/mu2e-celeritas.sif)
BASE=$(realpath /scratch/$USER/mu2efastsim/probe)
OUT=$BASE/celer_smoke_${SLURM_JOB_ID}
GPU=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)
[[ $GPU == *A100* ]] || { echo 'A100 required' >&2; exit 2; }
printf 'job=%s host=%s utc=%s\n%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" "$GPU"
stat -c '%n %s bytes %y' "$IMAGE"
sha256sum "$SCRIPT"
apptainer exec --cleanenv --nv --bind "$BASE:$BASE" --bind "$HERE:$HERE:ro" "$IMAGE" \
    python3 "$SCRIPT" "$OUT" --events 64 --field-tesla 1
