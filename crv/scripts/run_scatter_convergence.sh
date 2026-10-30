#!/usr/bin/env bash
# Run one bounded opt-in scatter-in study in an already configured Python environment.
set -euo pipefail
ROOT=${1:?reference root}
OUT=${2:?exclusive new output root}
STEP=${3:-12.5}
ENVELOPE=${4:-1500}
shift "$(( $# >= 4 ? 4 : $# ))"
python3 -m crvfastsim.analyses.scatter_convergence --reference-root "$ROOT" --outdir "$OUT" \
  --gdml "$ROOT/geometry/mu2e_run2_v10.gdml" --step-mm "$STEP" --envelope-mm "$ENVELOPE" "$@"
