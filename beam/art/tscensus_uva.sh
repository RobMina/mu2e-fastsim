#!/bin/bash
# Run a read-only MetaCat census on an allocated CPU node without FNAL credentials.
set -eo pipefail
fail() { echo "tscensus_uva: $*" >&2; exit 2; }
SELF=$(readlink -f "$0")
if [[ ${1:-} == --inside ]]; then
  shift
  CODE=$1; OUT=$2; shift 2
  source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
  setup muse
  muse setup ops
  export PYTHONPATH="$CODE/beam${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONDONTWRITEBYTECODE=1
  python3 -m beamfastsim.tscensus --out "$OUT" "$@"
  exit $?
fi
if [[ ${1:-} == --mounted ]]; then
  shift
  exec apptainer exec --userns --cleanenv --containall -B /cvmfs "$@"
fi
[[ -n ${SLURM_JOB_ID:-} ]] || fail 'a Slurm CPU allocation is required'
[[ $# -ge 2 ]] || fail 'usage: tscensus_uva.sh CODE_ROOT OUT_JSON [census arguments]'
CODE=$(readlink -f "$1"); OUT=$(readlink -m "$2"); shift 2
[[ -f $CODE/beam/beamfastsim/tscensus.py ]] || fail 'CODE_ROOT must contain beam/beamfastsim/tscensus.py'
[[ -x ${CVMFSEXEC_DIST:-}/cvmfsexec && -d $CVMFSEXEC_DIST/dist ]] || fail 'supply an existing CVMFSEXEC_DIST'
LOCAL=$(mktemp -d /tmp/tscensus-uva.XXXXXXXX)
trap 'rm -rf "$LOCAL"' EXIT
cp -a "$CVMFSEXEC_DIST" "$LOCAL/cvmfsexec"
CE="$LOCAL/cvmfsexec"
mkdir -p "$LOCAL/cache" "$CE/dist/etc/cvmfs/domain.d" "$(dirname "$OUT")"
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
IMAGE=/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9:latest
printf 'start=%s host=%s job=%s\n' "$(date -u +%FT%TZ)" "$(hostname)" "$SLURM_JOB_ID"
"$CE/cvmfsexec" config-osg.opensciencegrid.org mu2e.opensciencegrid.org fermilab.opensciencegrid.org singularity.opensciencegrid.org -- \
  /bin/bash "$SELF" --mounted -B /dev/fuse -B "$LOCAL:$LOCAL" -B "$CODE:$CODE" \
  -B "$(dirname "$OUT")" "$IMAGE" /bin/bash "$SELF" --inside "$CODE" "$OUT" "$@"
