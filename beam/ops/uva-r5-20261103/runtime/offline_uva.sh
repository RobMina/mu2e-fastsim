#!/bin/bash
# Run an already-built Offline area using a private node-local CVMFS client.
set -eo pipefail
fail() { echo "offline_uva: $*" >&2; exit 2; }
if [[ ${1:-} == --mounted ]]; then
  shift
  printf 'mounted_image=%s\n' "$(readlink -f "$OFFLINE_MOUNT_IMAGE")"
  exec apptainer exec --userns --cleanenv --containall -B /cvmfs "$@"
fi
if [[ ${1:-} == --inside ]]; then
  shift
  AREA=$1; FCL=$2; shift 2
  if [[ $(. /etc/os-release; echo "$ID:$VERSION_ID") == rocky:9* ]]; then
    export MU2E_OSNAME=al9 MU2E_SPACK=true UPS_OVERRIDE="-H Linux64bit+5.14-2.34-al9-3"
  fi
  source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
  setup muse
  if [[ -f $AREA/setup.sh && -d $AREA/backing ]]; then
    source "$AREA/setup.sh" || echo "setup.sh returned $?; checking resolved area and executable"
  else
    cd "$AREA"
    muse setup || echo "muse setup returned $?; checking resolved area and executable"
  fi
  [[ $(readlink -f "$MUSE_WORK_DIR") == "$(readlink -f "$AREA")" ]] || fail "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"
  if [[ -e $AREA/backing ]]; then
    BB="$(readlink -f "$AREA/backing")/build/$MUSE_STUB"
    [[ ! -d $BB ]] || export FHICL_FILE_PATH="$FHICL_FILE_PATH:$BB"
  fi
  [[ ! -d $AREA/build/$MUSE_STUB ]] || export FHICL_FILE_PATH="$FHICL_FILE_PATH:$AREA/build/$MUSE_STUB"
  export FHICL_FILE_PATH="$(dirname "$FCL"):${OFFLINE_FCL_PATH:+$OFFLINE_FCL_PATH:}$FHICL_FILE_PATH"
  [[ -z ${OFFLINE_SEARCH_PATH:-} ]] || export MU2E_SEARCH_PATH="$OFFLINE_SEARCH_PATH:$MU2E_SEARCH_PATH"
  command -v mu2e
  printf 'MUSE_WORK_DIR=%s MUSE_ENVSET=%s MUSE_STUB=%s\n' "$MUSE_WORK_DIR" "$MUSE_ENVSET" "$MUSE_STUB"
  cd "$OFFLINE_WORK_DIR"
  mu2e --version
  fhicl-dump -c "$FCL" > "${OFFLINE_LOG_PREFIX}.expanded.fcl"
  args=(-c "$FCL")
  [[ -z ${OFFLINE_OUTPUT_FILE:-} ]] || args+=(-T "$OFFLINE_OUTPUT_FILE")
  [[ -z ${OFFLINE_EVENTS:-} ]] || args+=(-n "$OFFLINE_EVENTS")
  [[ $# == 0 ]] || args+=(-s "$@")
  "$OFFLINE_TIME" -v -o "${OFFLINE_LOG_PREFIX}.art.time" mu2e "${args[@]}"
  if [[ ${OFFLINE_WARM_REPEAT:-0} == 1 ]]; then
    [[ -n ${OFFLINE_OUTPUT_FILE:-} ]] || fail 'OFFLINE_WARM_REPEAT needs OFFLINE_OUTPUT_FILE'
    args[3]="${OFFLINE_OUTPUT_FILE%.root}.warm.root"
    "$OFFLINE_TIME" -v -o "${OFFLINE_LOG_PREFIX}.warm.art.time" mu2e "${args[@]}"
  fi
  du -sh "${OFFLINE_CACHE_PATH:?}" | tee "${OFFLINE_LOG_PREFIX}.cache.txt"
  exit 0
fi
[[ $# -ge 2 ]] || fail 'usage: offline_uva.sh AREA_OR_TARBALL FCL [INPUT ...]'
[[ -n ${SLURM_JOB_ID:-} || ${OFFLINE_ALLOW_NONSLURM:-0} == 1 ]] || fail 'run in a compute-node Slurm allocation'
IMAGE=${OFFLINE_IMAGE:-/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9:latest}
[[ $IMAGE == /cvmfs/* || -e $IMAGE ]] || fail 'set OFFLINE_IMAGE to an EL9 image or CVMFS image directory'
[[ -x ${CVMFSEXEC_DIST:-}/cvmfsexec && -d $CVMFSEXEC_DIST/dist ]] || fail 'set CVMFSEXEC_DIST to a prepared cvmfsexec tree'
ROOT=${OFFLINE_NODE_TMP:-/tmp}
case $(readlink -f "$ROOT") in /tmp|/tmp/*|/localscratch|/localscratch/*|/local|/local/*) ;; *) fail 'CVMFS cache must be on node-local /tmp, /local or /localscratch';; esac
LOCAL=$(mktemp -d "$ROOT/offline-uva.XXXXXXXX")
trap 'rm -rf "$LOCAL"' EXIT
cp -a "$CVMFSEXEC_DIST" "$LOCAL/cvmfsexec"
cp /usr/bin/time "$LOCAL/time"
CE="$LOCAL/cvmfsexec"
mkdir -p "$LOCAL/cache" "$CE/dist/etc/cvmfs/domain.d"
cat > "$CE/dist/etc/cvmfs/domain.d/opensciencegrid.org.local" <<EOF
CVMFS_SERVER_URL="${OFFLINE_CVMFS_SERVERS:-http://cvmfs-s1fnal.opensciencegrid.org:8000/cvmfs/@fqrn@;http://cvmfs-s1bnl.opensciencegrid.org:8000/cvmfs/@fqrn@}"
EOF
cat > "$CE/dist/etc/cvmfs/default.local" <<EOF
CVMFS_HTTP_PROXY="${OFFLINE_CVMFS_PROXY:-DIRECT}"
CVMFS_CACHE_BASE=$LOCAL/cache
CVMFS_QUOTA_LIMIT=${OFFLINE_CACHE_MB:-6000}
CVMFS_TIMEOUT=30
CVMFS_TIMEOUT_DIRECT=30
EOF
if [[ $1 == /cvmfs/* ]]; then AREA=$1; else AREA=$(readlink -f "$1"); fi
FCL=$(readlink -f "$2"); shift 2
[[ -f $FCL ]] || fail "missing FCL: $FCL"
if [[ -f $AREA ]]; then
  mkdir "$LOCAL/area"
  tar -xjf "$AREA" -C "$LOCAL/area"
  AREA="$LOCAL/area/Code"
  [[ -f $AREA/setup.sh ]] || fail 'trusted muse tarball must contain Code/setup.sh'
elif [[ ! -d $AREA && $AREA != /cvmfs/* ]]; then
  fail "missing area: $AREA"
fi
WORK=$(pwd -P)
PREFIX=${OFFLINE_LOG_PREFIX:-offline}
[[ $PREFIX != */* ]] || fail 'OFFLINE_LOG_PREFIX must be a filename prefix, not a path'
SELF=$(readlink -f "$0")
binds=(-B /dev/fuse -B "$LOCAL:$LOCAL" -B "$WORK:$WORK" -B "$(dirname "$SELF")" -B "$(dirname "$FCL")")
[[ $AREA == /cvmfs/* || $AREA == "$LOCAL/"* ]] || binds+=(-B "$AREA:$AREA")
inputs=()
for input in "$@"; do
  input=$(readlink -f "$input")
  [[ -f $input ]] || fail "missing input: $input"
  inputs+=("$input")
  binds+=(-B "$(dirname "$input")")
done
[[ -z ${OFFLINE_BINDS:-} ]] || binds+=(-B "$OFFLINE_BINDS")
envs=(--env "OFFLINE_WORK_DIR=$WORK" --env "OFFLINE_LOG_PREFIX=$PREFIX" --env "OFFLINE_TIME=$LOCAL/time" --env "OFFLINE_CACHE_PATH=$LOCAL/cache")
[[ -z ${OFFLINE_OUTPUT_FILE:-} ]] || OFFLINE_OUTPUT_FILE=$(readlink -m "$OFFLINE_OUTPUT_FILE")
for key in OFFLINE_SEARCH_PATH OFFLINE_OUTPUT_FILE OFFLINE_EVENTS OFFLINE_FCL_PATH OFFLINE_WARM_REPEAT; do
  [[ -z ${!key:-} ]] || envs+=(--env "$key=${!key}")
done
printf 'start=%s host=%s job=%s image=%s\n' "$(date -u +%FT%TZ)" "$(hostname)" "${SLURM_JOB_ID:-none}" "$IMAGE"
if [[ $IMAGE == /cvmfs/* ]]; then
  export OFFLINE_MOUNT_IMAGE=$IMAGE
  /usr/bin/time -v -o "$WORK/$PREFIX.total.time" "$CE/cvmfsexec" \
    config-osg.opensciencegrid.org mu2e.opensciencegrid.org fermilab.opensciencegrid.org singularity.opensciencegrid.org -- \
    /bin/bash "$SELF" --mounted "${binds[@]}" "${envs[@]}" "$IMAGE" \
    /bin/bash "$SELF" --inside "$AREA" "$FCL" "${inputs[@]}"
else
  /usr/bin/time -v -o "$WORK/$PREFIX.total.time" apptainer exec --cleanenv --containall \
    "${binds[@]}" "${envs[@]}" "$IMAGE" \
    "$CE/cvmfsexec" config-osg.opensciencegrid.org mu2e.opensciencegrid.org fermilab.opensciencegrid.org -- \
    /bin/bash "$SELF" --inside "$AREA" "$FCL" "${inputs[@]}"
fi
