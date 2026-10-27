#!/bin/bash
# Recover current memberships and audit retained controls/exposure without repeating response.
set -euo pipefail
BASE=${1:?retained replay base}
CURRENT=${2:?current fast output}
REPLICA=${3:?independent fast output}
OUT=${4:?new audit directory}
PROBES=${5:?retained Cat probe directory}
COUNTERS=${6:?matching Run1A counters file}
mkdir "$OUT"
for B in block0 block1 block2_fullinventory; do
  python3 -m crvfastsim.veto.replay_recover --check "$CURRENT/$B/check.json" \
    --fast "$CURRENT/$B/fast.parquet" --input "$BASE/$B/mixed.root" --counters "$COUNTERS" \
    --outdir "$CURRENT/$B/recovery" > "$OUT/${B}_recovery.log" 2>&1
done
python3 -m crvfastsim.veto.replay_audit --base "$BASE" --candidate "$CURRENT" \
  --replica "$REPLICA" --out "$OUT/paired" > "$OUT/paired.log" 2>&1
python3 -m crvfastsim.veto.exposure_audit --base "$BASE" --probes "$PROBES" \
  --out "$OUT/exposure.json" > "$OUT/exposure.log" 2>&1
python3 -m pytest -q tests/test_veto*.py tests/test_coinc_attachment_cpu.py > "$OUT/tests.txt" 2>&1
cat "$OUT/paired.log"
tail -4 "$OUT/tests.txt"
