#!/bin/bash
#SBATCH --job-name=m2fs_celer_h6b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
# Execute new control-isolation fixtures with the unchanged h6 binary and runner.
# Submit with the account and log location from site.env, e.g.
#   sbatch -A "$SBATCH_ACCOUNT" --output="$CELERITAS_PROBE_OUT/%x-%j.out" container/celeritas_adapter/field_probe/run_sensitivity.sh
# CELERITAS_FIELD_SENSITIVITY_INPUT holds the prepared checker/, fixtures/ and snapshot.sha256.
set -eo pipefail
module load apptainer
set -u
: "${SLURM_JOB_ID:?Submit with sbatch}"
SRC=${CELERITAS_FIELD_PROBE_ROOT:?set CELERITAS_FIELD_PROBE_ROOT to the prepared field-probe snapshot (site.env)}/src
INPUT=${CELERITAS_FIELD_SENSITIVITY_INPUT:?set CELERITAS_FIELD_SENSITIVITY_INPUT to the prepared sensitivity snapshot (site.env)}
BUILD=${CELERITAS_FIELD_PROBE_BUILD:?set CELERITAS_FIELD_PROBE_BUILD to the field-probe build directory (site.env)}
H5=${CELERITAS_BUILD_ROOT:?set CELERITAS_BUILD_ROOT to the existing Celeritas build root holding install/ (site.env)}
IMAGE=${CELERITAS_SIF:?set CELERITAS_SIF to the existing Celeritas image (site.env)}
BASE=${CELERITAS_PROBE_OUT:?set CELERITAS_PROBE_OUT to the probe evidence directory (site.env)}
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
