#!/bin/bash
# Run a command in the Mu2e environment of the stock EventNtuple release (PyROOT + libeventntuple_dict), for the
# centuple write stage:  mu2e_env.sh COMMAND [ARGS...]
#   - where /cvmfs/mu2e.opensciencegrid.org is mounted (GPVM, EAF): setupmu2e-art + muse setup of the musing;
#   - elsewhere: container/offline/offline_uva.sh (cvmfsexec + the FNAL worker image; needs CVMFSEXEC_DIST, a Slurm
#     allocation and /dev/fuse), with the command as its OFFLINE_CHAIN script.
# CENTUPLE_MUSING overrides the musing (default AnalysisMDC2025 v02_02_02, the release of Mu2eBO ce_chain's nts step).
set -eo pipefail
MUSING=${CENTUPLE_MUSING:-/cvmfs/mu2e.opensciencegrid.org/Musings/AnalysisMDC2025/v02_02_02}
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ -d /cvmfs/mu2e.opensciencegrid.org/Musings ]]; then
  set +eu
  source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
  setup muse >/dev/null 2>&1 || true
  source "$MUSING/setup.sh"
  set -e
  exec "$@"
fi
RUNNER=${CENTUPLE_OFFLINE_RUNNER:-$HERE/../../../container/offline/offline_uva.sh}
[[ -f $RUNNER ]] || { echo "mu2e_env: no /cvmfs and no offline runner at $RUNNER" >&2; exit 2; }
WORK=$(pwd -P)
REPO=$(cd "$HERE/../../.." && pwd -P)
export OFFLINE_BINDS=${OFFLINE_BINDS:+$OFFLINE_BINDS,}$REPO
SCRIPT=$(mktemp "$WORK/.centuple_chain.XXXXXX.sh")
trap 'rm -f "$SCRIPT"' EXIT
{ echo 'set -eo pipefail'; printf 'cd %q\n' "$WORK"; printf '%q ' "$@"; echo; } > "$SCRIPT"
FCL=$(mktemp "$WORK/.centuple_fcl.XXXXXX.fcl")
trap 'rm -f "$SCRIPT" "$FCL"' EXIT
echo '# placeholder: the runner checks an FCL path; OFFLINE_CHAIN runs instead of mu2e' > "$FCL"
OFFLINE_CHAIN="$SCRIPT" OFFLINE_LOG_PREFIX=${OFFLINE_LOG_PREFIX:-centuple_write} bash "$RUNNER" "$MUSING" "$FCL"
