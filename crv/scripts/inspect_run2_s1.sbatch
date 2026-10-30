#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Inspect the staged source with existing tools and a private CVMFS cache, never transport.
set -eo pipefail
if [[ ${1:-} == --inside ]]; then
  shift
  AREA=$1; SOURCE=$2; FCL=$3; OUT=$4
  source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
  setup muse
  source "$AREA/setup.sh" || echo "setup returned $?; checking resolved area"
  [[ $(readlink -f "$MUSE_WORK_DIR") == "$(readlink -f "$AREA")" ]] || exit 2
  export FHICL_FILE_PATH="$(dirname "$FCL"):$FHICL_FILE_PATH"
  cd "$OUT"
  mu2e --version
  config_dumper --help > config_dumper.help.txt || [[ $? == 1 ]]
  config_dumper -P "$SOURCE" > embedded.fcl
  fhicl-dump -c "$FCL" > scope.expanded.fcl
  /usr/bin/time -v -o probe.time mu2e -c "$FCL" -s "$SOURCE" -n 1 > probe.log 2>&1
  /usr/bin/time -v -o scope.time mu2e -c "$FCL" -s "$SOURCE" > scope.log 2>&1
  exit 0
fi
: "${REUSE_CODE:?}" "${REUSE_ROOT:?}" "${PRODUCTION_INPUT:?}" "${PRODUCTION_SHA256:?}"
: "${OFFLINE_RUNNER:?}" "${OFFLINE_AREA:?}" "${CVMFSEXEC_DIST:?}" "${OFFLINE_IMAGE:?}"
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
module load apptainer
printf '%s  %s\n' "$PRODUCTION_SHA256" "$PRODUCTION_INPUT" | sha256sum -c -
OUT="$REUSE_ROOT/scope_${SLURM_JOB_ID}"
mkdir "$OUT"
OUT=$(readlink -f "$OUT")
LOCAL=$(mktemp -d /tmp/s4-scope.XXXXXXXX)
trap 'rm -rf "$LOCAL"' EXIT
cp -a "$CVMFSEXEC_DIST" "$LOCAL/cvmfsexec"
CE="$LOCAL/cvmfsexec"
mkdir -p "$LOCAL/cache" "$LOCAL/area" "$CE/dist/etc/cvmfs/domain.d"
cat > "$CE/dist/etc/cvmfs/domain.d/opensciencegrid.org.local" <<EOF
CVMFS_SERVER_URL="http://cvmfs-s1fnal.opensciencegrid.org:8000/cvmfs/@fqrn@;http://cvmfs-s1bnl.opensciencegrid.org:8000/cvmfs/@fqrn@"
EOF
cat > "$CE/dist/etc/cvmfs/default.local" <<EOF
CVMFS_HTTP_PROXY=DIRECT
CVMFS_CACHE_BASE=$LOCAL/cache
CVMFS_QUOTA_LIMIT=6000
CVMFS_TIMEOUT=30
CVMFS_TIMEOUT_DIRECT=30
EOF
tar -xjf "$OFFLINE_AREA" -C "$LOCAL/area"
AREA="$LOCAL/area/Code"
[[ -f "$AREA/setup.sh" ]] || exit 2
SELF=$(readlink -f "$0")
RUNNER=$(readlink -f "$OFFLINE_RUNNER")
FCL=$(readlink -f "$REUSE_CODE/art/CrvFastSim/fcl/inspect_run2_s1_scope.fcl")
SOURCE=$(readlink -f "$PRODUCTION_INPUT")
export OFFLINE_MOUNT_IMAGE="$OFFLINE_IMAGE"
"$CE/cvmfsexec" config-osg.opensciencegrid.org mu2e.opensciencegrid.org fermilab.opensciencegrid.org singularity.opensciencegrid.org -- \
  /bin/bash "$RUNNER" --mounted -B /dev/fuse -B "$LOCAL" -B "$OUT" \
  -B "$(dirname "$RUNNER")" -B "$(dirname "$SELF")" -B "$(dirname "$FCL")" -B "$(dirname "$SOURCE")" \
  "$OFFLINE_IMAGE" /bin/bash "$SELF" --inside "$AREA" "$SOURCE" "$FCL" "$OUT"
