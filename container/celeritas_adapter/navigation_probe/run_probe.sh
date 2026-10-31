#!/bin/bash
#SBATCH --job-name=m2fs_celer_h7a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
# Runtime only, after the user build has been independently verified.
# Submit with the account and log location from site.env, e.g.
#   sbatch -A "$SBATCH_ACCOUNT" --output="$CELERITAS_PROBE_OUT/%x-%j.out" container/celeritas_adapter/navigation_probe/run_probe.sh
# CELERITAS_NAV_PROBE_ROOT holds the prepared src/, fixtures/ and snapshot.sha256; CELERITAS_NAV_PROBE_BUILD the build.
set -eo pipefail
module load apptainer
set -u
: "${SLURM_JOB_ID:?Submit with sbatch}"
ROOT=${CELERITAS_NAV_PROBE_ROOT:?set CELERITAS_NAV_PROBE_ROOT to the prepared navigation-probe snapshot (site.env)}
BUILD=${CELERITAS_NAV_PROBE_BUILD:?set CELERITAS_NAV_PROBE_BUILD to the navigation-probe build directory (site.env)}
IMAGE=${CELERITAS_SIF:?set CELERITAS_SIF to the existing Celeritas image (site.env)}
H5=${CELERITAS_BUILD_ROOT:?set CELERITAS_BUILD_ROOT to the existing Celeritas build root holding install/ (site.env)}
BASE=${CELERITAS_PROBE_OUT:?set CELERITAS_PROBE_OUT to the probe evidence directory (site.env)}
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
  "$IMAGE" \
  python3 "$ROOT/src/run_probe.py" "$OUT/results" --source "$ROOT/src" --fixtures "$ROOT/fixtures" \
  --build "$BUILD" --prefix "$H5/install" --sanitizer
