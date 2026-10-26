#!/bin/bash
# Recheck retained configured/legacy-stock arms separately without running either response.
set -e
BASE=${1:?replay data root}
OUT=${2:?new report directory}
CONDITIONS=${3:?fast conditions dump}
SELECTION=${4:?selected-track parquet}
REFERENCE=${5:?saved deadtime stats}
shift 5
[ "$#" -gt 0 ] || { echo "give retained block names"; exit 2; }
[ -z "${MUSE_WORK_DIR:-}" ] || { echo "use a fresh ana-only shell"; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir "$OUT"
for policy in intended_per_sector_v1 legacy_stock_shared_lut_v1; do
  SUFFIX=
  [ "$policy" != legacy_stock_shared_lut_v1 ] || SUFFIX=_alias
  CHECKS=(); FAST=()
  for block in "$@"; do
    case "$block" in *[/]*|.|..) echo "block must be a basename"; exit 2;; esac
    O=$BASE/$block
    STOCK=$O/stock_retry/stock.root
    [ -f "$STOCK" ] || STOCK=$O/stock/stock.root
    CHECK=$OUT/${block}_${policy}_check.json
    python3 -m crvfastsim.veto.replay_check --input "$O/mixed.root" --stock "$STOCK" \
      --events "${VETO_REPLAY_EVENTS:-3000}" --conditions "$CONDITIONS" --fast "$O/fast${SUFFIX}.parquet" \
      --legacy-yield-policy "$policy" --out "$CHECK" > "$OUT/${block}_${policy}_check.log" 2>&1
    CHECKS+=("$CHECK"); FAST+=("$O/fast${SUFFIX}.parquet")
  done
  python3 -m crvfastsim.veto.replay_metrics --checks "${CHECKS[@]}" --fast "${FAST[@]}" \
    --selection "$SELECTION" --reference "$REFERENCE" --out "$OUT/paired_${policy}.json"
done
