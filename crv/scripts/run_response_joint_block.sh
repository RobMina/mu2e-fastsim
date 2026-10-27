#!/bin/bash
# Bounded two-by-two response diagnostic, with exact saved-state gates and frozen background/time calibration.
set -euo pipefail
DATA=${1:?validation root}
OUT=${2:?new output directory}
BLOCK=${3:?a or b}
case "$BLOCK" in
  a) seed=2026102501 ;;
  b) seed=2026102502 ;;
  *) echo 'block must be a or b' >&2; exit 2 ;;
esac
mkdir "$OUT"
sha256sum "$0" > "$OUT/driver.sha256"
printf '%s\n' '{"schema":1,"response":{"sectors":{"T1":{"yield_per_mev":20027.0}}}}' > "$OUT/yield_070.json"
python3 -m crvfastsim.objective.validation run --manifest "$DATA/heldout_$BLOCK.json" \
  --out "$OUT/nominal" --seed "$seed" --calibration --diagnostic-code --joint-capture
python3 -m crvfastsim.objective.validation compare-replay \
  --reference "$DATA/replicas_cpu_v1/${BLOCK}_nominal/r3.json" --candidate "$OUT/nominal/r3.json" \
  --out "$OUT/nominal_capture_gate.json"
python3 -m crvfastsim.objective.validation run --manifest "$DATA/heldout_$BLOCK.json" \
  --out "$OUT/atten" --seed "$seed" --calibration --diagnostic-code --joint-capture \
  --knob atten_T1=0.4367132104009329
python3 -m crvfastsim.objective.validation compare-replay \
  --reference "$DATA/controls_${BLOCK}_cpu_v1/${BLOCK}_t1atten/r3.json" --candidate "$OUT/atten/r3.json" \
  --out "$OUT/atten_capture_gate.json"
python3 -m crvfastsim.objective.validation run --manifest "$DATA/heldout_$BLOCK.json" \
  --out "$OUT/yield" --seed "$seed" --calibration --diagnostic-code --joint-capture \
  --response-config "$OUT/yield_070.json"
python3 -m crvfastsim.objective.validation run --manifest "$DATA/heldout_$BLOCK.json" \
  --out "$OUT/joint" --seed "$seed" --calibration --diagnostic-code --joint-capture \
  --response-config "$OUT/yield_070.json" --knob atten_T1=0.4367132104009329
printf 'complete\n' > "$OUT/complete.txt"
