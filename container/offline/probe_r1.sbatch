#!/bin/bash
#SBATCH --account=ldmxuva
#SBATCH --partition=standard
#SBATCH --job-name=offline_r1_probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -eo pipefail
ROOT=${OFFLINE_ROOT:?set OFFLINE_ROOT to your output directory}
module load apptainer
LOCAL=$(mktemp -d /tmp/offline-r1.XXXXXXXX)
trap 'rm -rf "$LOCAL"' EXIT
OUT="$ROOT/probe_${SLURM_JOB_ID}"
mkdir -p "$OUT"
if [[ -n ${CVMFSEXEC_DIST:-} ]]; then
  cp -a "$CVMFSEXEC_DIST" "$LOCAL/cvmfsexec"
else
  git clone https://github.com/cvmfs/cvmfsexec.git "$LOCAL/cvmfsexec" > "$OUT/clone.log" 2>&1
  git -C "$LOCAL/cvmfsexec" rev-parse HEAD > "$OUT/cvmfsexec.commit"
  (cd "$LOCAL/cvmfsexec" && ./makedist -m rhel9-x86_64 osg) > "$OUT/makedist.log" 2>&1
  cp -a "$LOCAL/cvmfsexec" "$OUT/distribution"
fi
cp /usr/bin/time "$LOCAL/time"
/usr/bin/time -v -o "$OUT/container.time" apptainer exec --cleanenv --containall \
  -B /dev/fuse -B "$ROOT:$ROOT" -B "$LOCAL:$LOCAL" \
  "${OFFLINE_IMAGE:?set OFFLINE_IMAGE to an existing EL9 SIF}" \
  /bin/bash "$ROOT/code/container/offline/probe_r1.sh" "$OUT" "$LOCAL"
