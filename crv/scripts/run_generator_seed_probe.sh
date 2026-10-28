#!/bin/bash
# Initialize the historical stock producer with zero events and print seed assignments.
set -eo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CORE=${M2FS_CORE:?read-only core snapshot}
if ! declare -F pyenv >/dev/null; then unset MU2E; fi
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana
ANA_PYTHON=$(command -v python3)
AREA=/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay
cd "$AREA"
muse setup
test "$MUSE_WORK_DIR" = "$AREA" || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
export PYTHONPATH="$CODE:$CORE" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$ANA_PYTHON" -m valhist.generator_seed_probe "$@"
