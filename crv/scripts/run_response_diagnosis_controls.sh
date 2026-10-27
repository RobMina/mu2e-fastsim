#!/bin/bash
# Two single-knob interventions per held-out block, reusing the original frozen runtime and nominal outputs.
set -euo pipefail
DATA=${1:?validation directory with heldout manifests and replicas_cpu_v1}
OUT=${2:?new output directory}
BLOCK=${3:?a or b}
case "$BLOCK" in
  a) seed=2026102501 ;;
  b) seed=2026102502 ;;
  *) echo 'block must be a or b' >&2; exit 2 ;;
esac
mkdir "$OUT"
sha256sum "$0" > "$OUT/driver.sha256"
python3 -m crvfastsim.objective.validation run --manifest "$DATA/heldout_$BLOCK.json" \
  --out "$OUT/${BLOCK}_t1atten" --seed "$seed" --calibration --knob atten_T1=0.4367132104009329
python3 -m crvfastsim.objective.validation run --manifest "$DATA/heldout_$BLOCK.json" \
  --out "$OUT/${BLOCK}_exspeed" --seed "$seed" --calibration --knob speed_EX=0.9840676938698677
printf 'complete\n' > "$OUT/complete.txt"
