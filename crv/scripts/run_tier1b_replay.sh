#!/bin/bash
# Replay retained steps in the certified frozen stock area, without rebuilding or regenerating transport.
set -eo pipefail
OUT=${1:?prepared output}; AREA=${2:?frozen stock area}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CORE=${M2FS_CORE:?read-only core snapshot}
if ! declare -F pyenv >/dev/null; then unset MU2E; fi
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana
export PYTHONPATH="$CODE:$CORE" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
D=${CRVFS_DATA:?data root}
export CRVFS_LOOKUP_DIR="$D/lookup_v6_0"
STEPS=$(jq -r .steps "$OUT/manifest.json")
SEED=$(jq -r .seed "$OUT/manifest.json")
GEN=$(jq -r .gen_art "$OUT/manifest.json")
CALIB=$(jq -r .calib "$OUT/manifest.json")
DEV=$(jq -r .deviation "$OUT/manifest.json")
test "$(jq -r .area "$OUT/manifest.json")" = "$AREA" || exit 2
test ! -e "$OUT/tables/chain.root" || { echo 'refusing existing replay'; exit 2; }
python3 -m valhist.fasttables chain "$STEPS" "$OUT/tables/chain.root" --calib "$CALIB" --deviation "$DEV" \
  --seed "$SEED" > "$OUT/logs/fasttables.log" 2>&1
(
  cd "$AREA"
  muse setup
  test "$MUSE_WORK_DIR" = "$AREA" || exit 2
  export FHICL_FILE_PATH="$OUT:$FHICL_FILE_PATH"
  for arm in a1 e2e; do
    cd "$OUT/$arm"
    if mu2e -c "$arm.fcl" -s "$GEN" -T valhist.root > "$OUT/logs/$arm.log" 2>&1; then
      echo 0 > "$OUT/logs/$arm.exit"
    else
      rc=$?; echo "$rc" > "$OUT/logs/$arm.exit"; exit "$rc"
    fi
  done
)
python3 -m valhist.tier1b_replay verify "$OUT"
