#!/bin/bash
# Re-dump a planned donor interval with the user-built strict dumper; one sequential job per host.
set -eo pipefail
PLAN=${1:?plan.json}; FIRST=${2:-1}; LAST=${3:-}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CORE=${M2FS_CORE:?core checkout}
if ! declare -F pyenv >/dev/null; then unset MU2E; fi
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
pyenv ana
ANA_PYTHON=$(command -v python3)
PLAN=$(cd "$(dirname "$PLAN")" && printf '%s/%s' "$PWD" "$(basename "$PLAN")")
cd "$CODE/art"
muse setup
test "$MUSE_WORK_DIR" = "$CODE/art" || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR" >&2; exit 2; }
export PYTHONPATH="$CODE:$CORE" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
args=(redump "$PLAN" --first "$FIRST")
if [ -n "$LAST" ]; then args+=(--last "$LAST"); fi
exec "$ANA_PYTHON" -m valhist.genassoc_campaign "${args[@]}"
