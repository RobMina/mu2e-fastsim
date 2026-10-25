#!/bin/bash
# Check retained stock/fast outputs and compute paired metrics in an ana-only shell, without rerunning response.
set -e
BASE=${1:?replay data root}
CONDITIONS=${2:?fast conditions dump}
SELECTION=${3:?selected-track parquet}
REFERENCE=${4:?saved deadtime stats}
shift 4
[ "$#" -gt 0 ] || { echo "give block directory names after the four arguments"; exit 2; }
[ -z "${MUSE_WORK_DIR:-}" ] || { echo "start a fresh pyenv ana session without muse setup"; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CHECKS=(); FAST=()
for b in "$@"; do
  O=$BASE/$b
  STOCK=$O/stock_retry/stock.root
  [ -f "$STOCK" ] || STOCK=$O/stock/stock.root
  if [ ! -f "$O/check.json" ]; then
    python3 -m crvfastsim.veto.replay_check --input "$O/mixed.root" --stock "$STOCK" --events "${VETO_REPLAY_EVENTS:-3000}" \
      --conditions "$CONDITIONS" --fast "$O/fast.parquet" --out "$O/check.json" > "$O/check_ana.log" 2>&1
  fi
  CHECKS+=("$O/check.json"); FAST+=("$O/fast.parquet")
done
python3 -m crvfastsim.veto.replay_metrics --checks "${CHECKS[@]}" --fast "${FAST[@]}" \
  --selection "$SELECTION" --reference "$REFERENCE" --out "${VETO_METRICS_OUT:-$BASE/paired_blocks.json}"
