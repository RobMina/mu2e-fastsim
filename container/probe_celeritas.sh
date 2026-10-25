#!/bin/bash
# Inspect the existing image inside an allocated A100 step; never build or transport events.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: bash $0 IMAGE.sif NEW_OUTPUT_DIR (inside an allocated Slurm A100 step)" >&2
    exit 2
fi
if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_STEP_ID:-} ]]; then
    echo "Refusing to run outside a Slurm job step; use srun within an existing allocation." >&2
    exit 2
fi
IMAGE=$(realpath "$1")
[[ -r $IMAGE && -f $IMAGE ]] || { echo "Image is not a readable file: $IMAGE" >&2; exit 2; }
for tool in apptainer nvidia-smi; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 2; }
done
GPU=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)
[[ $GPU == *A100* ]] || { echo "No A100 visible; this image targets sm_80." >&2; exit 2; }
mkdir -- "$2"
OUT=$(cd "$2" && pwd)
exec > >(tee "$OUT/probe.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code.txt"' EXIT
printf 'scope=load-only; no transport or physics validation\n'
printf 'utc=%s\nhost=%s\njob=%s\nstep=%s\nimage=%s\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$IMAGE"
stat -c 'image_bytes=%s image_mtime=%y' "$IMAGE"
printf '%s\n' "$GPU" | tee "$OUT/gpu.txt"
apptainer --version
for option in --help --version --dump-default; do
    name=${option#--}
    echo "Inspecting celer-g4 $option"
    apptainer exec --cleanenv --nv "$IMAGE" env \
        -u CELER_DISABLE -u CELER_DISABLE_DEVICE -u CELER_KILL_OFFLOAD \
        celer-g4 "$option" > "$OUT/$name.txt" 2> "$OUT/$name.stderr.txt"
done
echo 'PASS: binary loading and CLI inspection only; GPU transport remains untested.'
