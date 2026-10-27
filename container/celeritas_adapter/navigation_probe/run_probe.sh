#!/bin/bash
#SBATCH --job-name=m2fs_celer_h7a
#SBATCH --account=ldmxuva
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/mu2efastsim/probe/celer-h7a-%j.out
# Runtime only, after the user build has been independently verified.
set -eo pipefail
module load apptainer
set -u
: "${SLURM_JOB_ID:?Submit with sbatch}"
ROOT=/home/$USER/mu2e/fastsim_uva/celer_navigation_h7a
BUILD=/scratch/$USER/mu2efastsim/celer_navigation_h7a-build
H5=/scratch/$USER/mu2efastsim/celer_rdc_h5
BASE=/scratch/$USER/mu2efastsim/probe
OUT=$BASE/celer_navigation_h7a_${SLURM_JOB_ID}
GPU=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)
[[ $GPU == *A100* ]] || { echo 'A100 required' >&2; exit 2; }
ulimit -c 0
(cd "$ROOT" && sha256sum --check snapshot.sha256)
(cd "$ROOT/src" && sha256sum --check source.sha256)
(cd "$H5/install/include" && sha256sum --check "$ROOT/src/api-pins.sha256")
mkdir "$OUT"
printf 'job=%s host=%s utc=%s\n%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" "$GPU" | tee "$OUT/allocation.txt"
cp -r "$ROOT/src" "$OUT/replay-source"
cp "$ROOT/snapshot.sha256" "$OUT/"
srun --ntasks=1 apptainer exec --cleanenv --nv \
  --bind "$ROOT:$ROOT:ro" --bind "$BUILD:$BUILD:ro" --bind "$H5:$H5:ro" --bind "$BASE:$BASE" \
  /scratch/$USER/mu2efastsim/mu2e-celeritas.sif \
  python3 "$ROOT/src/run_probe.py" "$OUT/results" --source "$ROOT/src" --fixtures "$ROOT/fixtures" \
  --build "$BUILD" --prefix "$H5/install" --sanitizer
