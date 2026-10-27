#!/bin/bash
# Reuse fixed-stock products and run only the fast response in a new three-block tag.
set -euo pipefail
BASE=${1:?retained replay base}
OUT=${2:?new output directory}
CONDITIONS=${3:?conditions ROOT file}
PHOTON=${4:?photon deviation file}
SELECTION=${5:?selected tracks}
REFERENCE=${6:?frozen deadtime reference}
SEED=${7:-8100}
CODE=$(cd "$(dirname "$0")/.." && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CODE:${PYTHONPATH:?core must be on PYTHONPATH}"
BLOCKS=(block0 block1 block2_fullinventory)
for B in "${BLOCKS[@]}"; do
  test -r "$BASE/$B/mixed.root"
  test -r "$BASE/fixed_stock_v13_41_00_20261025/$B/stock.root"
done
mkdir "$OUT"
sha256sum "$CODE"/crvfastsim/*.py "$CODE"/crvfastsim/{veto,response,reco}/*.py "$0" > "$OUT/code_sha256.txt"
CHECKS=(); FAST=()
for I in 0 1 2; do
  B=${BLOCKS[$I]}
  mkdir "$OUT/$B"
  /usr/bin/time -p python3 -m crvfastsim.veto.replay_fast "$BASE/$B/mixed.root" \
    --conditions "$CONDITIONS" --photon-dev "$PHOTON" --seed "$((SEED + I))" --chunk 100 \
    --out "$OUT/$B/fast.parquet" --pulses "$OUT/$B/pulses.parquet" > "$OUT/$B/fast.log" 2>&1
  python3 -m crvfastsim.veto.replay_check --input "$BASE/$B/mixed.root" \
    --stock "$BASE/fixed_stock_v13_41_00_20261025/$B/stock.root" --events 3000 \
    --conditions "$CONDITIONS" --fast "$OUT/$B/fast.parquet" --out "$OUT/$B/check.json" > "$OUT/$B/check.log" 2>&1
  CHECKS+=("$OUT/$B/check.json"); FAST+=("$OUT/$B/fast.parquet")
  tail -3 "$OUT/$B/fast.log"
done
python3 -m crvfastsim.veto.replay_metrics --checks "${CHECKS[@]}" --fast "${FAST[@]}" \
  --selection "$SELECTION" --reference "$REFERENCE" --out "$OUT/paired.json"
sha256sum -c "$OUT/code_sha256.txt" > "$OUT/code_verified.txt"
echo 'Completed fast-only replay; stock products reused.'
