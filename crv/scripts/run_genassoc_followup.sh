#!/bin/bash
# Complete the immutable donor audit, library reconstruction and held-out replay in one sequential job.
set -eo pipefail
ROOT=$(cd "${1:?campaign directory}" && pwd)
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CORE=${M2FS_CORE:?core checkout}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
pyenv ana
export PYTHONPATH="$CODE:$CORE" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"
trap 'result=$?; printf "%s\n" "$result" > "$ROOT/logs/followup.exit"' EXIT
python3 -m valhist.genassoc_report donors "$ROOT/plan.json" --out "$ROOT/donor_summary.json"
python3 -m valhist.genassoc_campaign rebuild "$ROOT/plan.json"
python3 -m valhist.genassoc_report library-invariants "$ROOT/plan.json" --out "$ROOT/library/muon_library_guard.json"
python3 -m valhist.stepclosure shower-audit "$ROOT/library/library.json" --out "$ROOT/library/association_audit.json"
bash scripts/validate_genassoc.sh "$ROOT" 11-20
python3 -m valhist.genassoc_report closure "$ROOT" \
  --baseline "${STEP_LAYERS_OUT:?set STEP_LAYERS_OUT to the step-layer study directory}/takeover" --out "$ROOT/closure_summary.json"
