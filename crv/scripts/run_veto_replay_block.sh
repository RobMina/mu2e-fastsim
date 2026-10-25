#!/bin/bash
# Serial bounded replay block on a user-built area; source tables are staged only for the export.
set -e
AREA=${1:?built crv/art area}
OUT=${2:?new block output directory}
BLOCK=${3:?subset block index}
EVENTS=${4:-3000}
: "${CRVFS_CODE:?source site.env}"
: "${CRVFS_DATA:?source site.env}"
: "${VETO_CAT_INVENTORY:?completed basename inventory}"
: "${VETO_CAT_PNFS:?Cat parquet output root}"
: "${VETO_CONDITIONS:?conditions dump}"
: "${VETO_PHOTON_DEV:?photon deviation file}"
: "${ANA_PY:?ana python3 path}"
[ "${MUSE_WORK_DIR:-}" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR"; exit 2; }
case "$OUT" in /exp/mu2e/data/users/*) ;; *) echo "output must be under data area"; exit 2;; esac
mkdir "$OUT"
export PATH="$(dirname "$ANA_PY"):$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRVFS_LOOKUP_DIR=/cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0
export MU2E_SEARCH_PATH="/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00:$MU2E_SEARCH_PATH"
SUBSET_ARGS=()
if [ -n "${VETO_SUBSET_EXCLUDE:-}" ]; then IFS=, read -r -a SUBSET_ARGS <<< "$VETO_SUBSET_EXCLUDE"; SUBSET_ARGS=(--exclude "${SUBSET_ARGS[@]}"); fi
"$ANA_PY" -m crvfastsim.veto.replay_subset "$VETO_CAT_INVENTORY" --pnfs "$VETO_CAT_PNFS" \
  --per-species 4 --block "${VETO_SUBSET_BLOCK:-$BLOCK}" --seed 20261025 --out "$OUT/subset.txt" "${SUBSET_ARGS[@]}"
bash "$CRVFS_CODE/scripts/export_veto_replay_subset.sh" "$OUT/subset.txt" "$OUT/mixed.root" "$EVENTS" "$((26000 + BLOCK))" > "$OUT/export.log" 2>&1
"$ANA_PY" -m crvfastsim.veto.replay_fast "$OUT/mixed.root" --conditions "$VETO_CONDITIONS" \
  --photon-dev "$VETO_PHOTON_DEV" --seed "$((8100 + BLOCK))" --chunk 100 \
  --out "$OUT/fast.parquet" --pulses "$OUT/pulses.parquet" > "$OUT/fast.log" 2>&1
bash "$CRVFS_CODE/scripts/run_veto_stock_replay.sh" "$AREA" "$OUT/mixed.root" "$OUT/stock" "$EVENTS" "$((9100 + BLOCK))" > "$OUT/stock_driver.log" 2>&1
cat "$OUT/fast.log"
cat "$OUT/stock_driver.log"
echo "Finished response block $BLOCK; run replay_check in a separate ana-only session: $OUT"
