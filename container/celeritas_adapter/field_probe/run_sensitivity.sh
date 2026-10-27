#!/bin/bash
#SBATCH --job-name=m2fs_celer_h6b
#SBATCH --account=ldmxuva
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/%u/mu2efastsim/probe/celer-h6b-%j.out
# Execute new control-isolation fixtures with the unchanged h6 binary and runner.
set -eo pipefail
module load apptainer
set -u
: "${SLURM_JOB_ID:?Submit with sbatch}"
SRC=/home/$USER/mu2e/fastsim_uva/celer_field_h6/src
INPUT=/home/$USER/mu2e/fastsim_uva/celer_field_h6b
BUILD=/scratch/$USER/mu2efastsim/celer_field_h6-build
H5=/scratch/$USER/mu2efastsim/celer_rdc_h5
IMAGE=/scratch/$USER/mu2efastsim/mu2e-celeritas.sif
BASE=/scratch/$USER/mu2efastsim/probe
OUT=$BASE/celer_field_h6b_${SLURM_JOB_ID}
GPU=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)
[[ $GPU == *A100* ]] || { echo 'A100 required' >&2; exit 2; }
ulimit -c 0
mkdir "$OUT"
printf 'job=%s host=%s utc=%s\n%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" "$GPU" | tee "$OUT/allocation.txt"
printf '%s  %s\n' \
  7b6c29280c469ac98a21c5ec26e352c96c78734f115b16e36b70c28f133c78bf \
  "$SRC/probe-manifest.json" \
  a423c659bda77c78b738fc50c5a63540caecf1807aa19f16beb7a395e10922a6 \
  "$BUILD/m2fs-field-probe" | sha256sum --check
(cd "$INPUT" && sha256sum --check snapshot.sha256)
cp -r "$INPUT/checker" "$OUT/replay-source"
cp "$INPUT/snapshot.sha256" "$OUT/"
sha256sum "$0" "$BUILD/m2fs-field-probe" | tee "$OUT/runtime.sha256"
srun --ntasks=1 apptainer exec --cleanenv --nv \
  --bind "$BASE:$BASE" --bind "$SRC:$SRC:ro" --bind "$INPUT:$INPUT:ro" \
  --bind "$BUILD:$BUILD:ro" --bind "$H5:$H5:ro" "$IMAGE" \
  python3 "$SRC/run_probe.py" "$OUT/results" --source "$SRC" --build "$BUILD" \
    --fixtures "$INPUT/fixtures" --prefix "$H5/install" --sanitizer
