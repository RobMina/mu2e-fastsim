#!/bin/bash
# Released single-foil diagnostic using the validated private-cache Offline UVA route.
set -eo pipefail
fail() { echo "edge_offline_uva: $*" >&2; exit 2; }
if [[ ${1:-} == --inside ]]; then
  if [[ $(. /etc/os-release; echo "$ID:$VERSION_ID") == rocky:9* ]]; then
    export MU2E_OSNAME=al9 MU2E_SPACK=true UPS_OVERRIDE="-H Linux64bit+5.14-2.34-al9-3"
  fi
  source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
  setup muse
  cd /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac
  muse setup || echo "muse setup returned $?; checking resolved release"
  [[ $MUSE_WORK_DIR == /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac ]] || fail 'wrong release'
  export FHICL_FILE_PATH="$EDGE_OUT:$MUSE_WORK_DIR/build/$MUSE_STUB:$FHICL_FILE_PATH"
  export MU2E_SEARCH_PATH="$EDGE_OUT:$MU2E_SEARCH_PATH"
  cd "$EDGE_OUT"
  command -v mu2e
  printf 'MUSE_WORK_DIR=%s MUSE_STUB=%s\n' "$MUSE_WORK_DIR" "$MUSE_STUB" > runtime.txt
  mu2e --version >> runtime.txt 2>&1
  sha256sum "$(readlink -f "$(command -v mu2e)")" > runtime.sha256
  OFF=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v10_47_00/Offline
  for path in Mu2eG4/src/ConstructEnvBox_tool.cc Mu2eG4/src/Mu2eStudyWorld.cc Mu2eG4/src/Mu2eG4Cuts.cc Analyses/src/Mu2eG4StudyReadBack_module.cc Mu2eG4/fcl/prolog.fcl; do
    sha256sum "$OFF/$path" >> runtime.sha256
  done
  fhicl-dump -c job.fcl > expanded.fcl
  "$EDGE_TIME" -v -o art.time mu2e -c job.fcl > job.log 2>&1
  du -sh "$EDGE_CACHE" > cache.txt
  exit 0
fi
[[ -n ${SLURM_JOB_ID:-} ]] || fail 'compute-node Slurm allocation required'
[[ $# == 1 && -d $1 ]] || fail 'usage: edge_offline_uva.sh PREPARED_OUTPUT_DIRECTORY'
[[ -x ${CVMFSEXEC_DIST:-}/cvmfsexec && -d $CVMFSEXEC_DIST/dist ]] || fail 'prepared EL8 client required'
module load apptainer
OUT=$(readlink -f "$1")
SELF=$(readlink -f "$0")
IMAGE=${OFFLINE_IMAGE:?set OFFLINE_IMAGE to the pinned stock EL9 CVMFS image}
[[ $IMAGE == /cvmfs/singularity.opensciencegrid.org/.images/sha256:* ]] || fail 'pinned CVMFS worker image required'
LOCAL=$(mktemp -d /tmp/beam-edge-uva.XXXXXXXX)
trap 'rm -rf "$LOCAL"' EXIT
cp -a "$CVMFSEXEC_DIST" "$LOCAL/cvmfsexec"
cp /usr/bin/time "$LOCAL/time"
CE="$LOCAL/cvmfsexec"
mkdir -p "$LOCAL/cache" "$CE/dist/etc/cvmfs/domain.d"
cat > "$CE/dist/etc/cvmfs/domain.d/opensciencegrid.org.local" <<'EOF'
CVMFS_SERVER_URL="http://cvmfs-s1fnal.opensciencegrid.org:8000/cvmfs/@fqrn@;http://cvmfs-s1bnl.opensciencegrid.org:8000/cvmfs/@fqrn@"
EOF
cat > "$CE/dist/etc/cvmfs/default.local" <<EOF
CVMFS_HTTP_PROXY=DIRECT
CVMFS_CACHE_BASE=$LOCAL/cache
CVMFS_QUOTA_LIMIT=6000
CVMFS_TIMEOUT=30
CVMFS_TIMEOUT_DIRECT=30
EOF
export OFFLINE_MOUNT_IMAGE=$IMAGE
printf 'start=%s host=%s job=%s image=%s\n' "$(date -u +%FT%TZ)" "$(hostname)" "$SLURM_JOB_ID" "$IMAGE" > "$OUT/launcher.txt"
set +e
/usr/bin/time -v -o "$OUT/total.time" "$CE/cvmfsexec" config-osg.opensciencegrid.org mu2e.opensciencegrid.org fermilab.opensciencegrid.org singularity.opensciencegrid.org -- \
  /bin/bash "${EDGE_MOUNT_RUNNER:?set EDGE_MOUNT_RUNNER to the unchanged Offline UVA wrapper}" --mounted \
  -B /dev/fuse -B "$LOCAL:$LOCAL" -B "$OUT:$OUT" -B "$(dirname "$SELF")" \
  --env "EDGE_OUT=$OUT" --env "EDGE_TIME=$LOCAL/time" --env "EDGE_CACHE=$LOCAL/cache" \
  "$IMAGE" /bin/bash "$SELF" --inside > "$OUT/launcher.log" 2>&1
result=$?
printf '%s\n' "$result" > "$OUT/exit_code.txt"
exit "$result"
