#!/bin/bash
# Reuse a corrected transport table for a fresh A1/E2E pilot in the frozen stock area, one shard on one host.
# Usage: run_pilot.sh NEW_OUT FASTC LIBRARY GEN_ART SEED; CRVVAL_AREA and M2FS_CORE name existing frozen areas.
set -e
OUT=${1:?new output directory}; FASTC=${2:?existing fastC table}; LIB=${3:?library}
GEN=${4:?gen art input}; SEED=${5:?seed}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AREA=${CRVVAL_AREA:?frozen stock area}; CORE=${M2FS_CORE:?isolated core checkout}
D=${CRVFS_DATA:?data root}
if ! declare -F pyenv >/dev/null; then unset MU2E; fi
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana
export PYTHONPATH="$CODE:$CORE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CRVFS_LOOKUP_DIR="$D/lookup_v6_0"
python3 -m valhist.pilot prepare --out "$OUT" --fastc "$FASTC" --library "$LIB" --area "$AREA" \
  --core "$CORE" --gen-art "$GEN" --seed "$SEED"
python3 -m crvfastsim tier1b-apply "$FASTC" --library "$LIB" --outdir "$OUT/tables" --seed "$SEED" \
  > "$OUT/logs/tier1b.log" 2>&1
TABLE="$OUT/tables/$(basename "$FASTC" | sed 's/^fastC\./fast1b./')"
python3 -m valhist.fasttables chain "$TABLE" "$OUT/tables/chain.root" \
  --calib "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv" \
  --deviation "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" --seed "$SEED" \
  > "$OUT/logs/fasttables.log" 2>&1
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
python3 -m valhist.pilot verify "$OUT"
