#!/bin/bash
#SBATCH --job-name=m2fs_celer_h6
#SBATCH --account=ldmxuva
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/%u/mu2efastsim/probe/celer-h6-%j.out
# Execute the immutable h6 probe under memcheck without compiling anything.
set -eo pipefail
module load apptainer
set -u
: "${SLURM_JOB_ID:?Submit with sbatch}"
ROOT=/home/$USER/mu2e/fastsim_uva/celer_field_h6
BUILD=/scratch/$USER/mu2efastsim/celer_field_h6-build
H5=/scratch/$USER/mu2efastsim/celer_rdc_h5
IMAGE=/scratch/$USER/mu2efastsim/mu2e-celeritas.sif
BASE=/scratch/$USER/mu2efastsim/probe
EVIDENCE=$BASE/celer_field_h6_${SLURM_JOB_ID}
GPU=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)
[[ $GPU == *A100* ]] || { echo 'A100 required' >&2; exit 2; }
ulimit -c 0
mkdir "$EVIDENCE"
printf 'job=%s host=%s utc=%s\n%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" "$GPU" | tee "$EVIDENCE/allocation.txt"
printf '%s  %s\n' \
  7b6c29280c469ac98a21c5ec26e352c96c78734f115b16e36b70c28f133c78bf \
  "$ROOT/src/probe-manifest.json" | sha256sum --check
sha256sum "$0" "$BUILD/m2fs-field-probe" "$ROOT/src/probe-manifest.json" | tee "$EVIDENCE/provenance.sha256"
cp "$BUILD/configure.log" "$BUILD/build.log" "$BUILD/CMakeCache.txt" \
  "$BUILD/compile_commands.json" "$ROOT/src/probe-manifest.json" "$EVIDENCE/"
srun --ntasks=1 apptainer exec --cleanenv --nv \
  --bind "$BASE:$BASE" --bind "$ROOT:$ROOT:ro" --bind "$BUILD:$BUILD:ro" --bind "$H5:$H5:ro" \
  "$IMAGE" python3 "$ROOT/src/run_probe.py" "$EVIDENCE/results" \
    --source "$ROOT/src" --build "$BUILD" --fixtures "$ROOT/fixtures" \
    --prefix "$H5/install" --sanitizer
