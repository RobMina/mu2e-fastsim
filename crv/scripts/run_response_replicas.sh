#!/bin/bash
# Four serial, one-pass evaluations on the certified disjoint held-out A/B manifests.
set -euo pipefail
DATA=${1:?validation data directory holding heldout_a.json and heldout_b.json}
OUT=${2:?new output directory}
mkdir "$OUT"
BEST=(speed_EX=0.9840676938698677 atten_EX=0.9826665625747099 atten_T1=0.4367132104009329 speed_T2=0.9730162915911263 atten_T2=1.1530157699497805)
for block in a b; do
  seed=2026102501; [ "$block" = b ] && seed=2026102502
  for arm in nominal best; do
    knobs=(); [ "$arm" = best ] && knobs=(--knob "${BEST[@]}")
    python3 -m crvfastsim.objective.validation run --manifest "$DATA/heldout_$block.json" \
      --out "$OUT/${block}_${arm}" --seed "$seed" --calibration "${knobs[@]}"
  done
done
printf 'complete\n' > "$OUT/complete.txt"
