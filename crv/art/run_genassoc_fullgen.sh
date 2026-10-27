#!/bin/bash
# Audit full-generator event census using the existing user-built isolated dumper.
set -eo pipefail
PLAN=${1:?production plan}; HISTORY=${2:?historical source sidecar}; OUT=${3:?new audit directory}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CORE=${M2FS_CORE:?core checkout}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
pyenv ana
ANA_PYTHON=$(command -v python3)
cd "$CODE/art"
muse setup
[ "$MUSE_WORK_DIR" = "$CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
export PYTHONPATH="$CODE:$CORE" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$ANA_PYTHON" -m valhist.genassoc_fullgen "$PLAN" --historical-source "$HISTORY" --outdir "$OUT"
