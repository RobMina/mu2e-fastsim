#!/bin/bash
# One complete P4 file supplies measured joint diagnostics without changing the frozen 5M-event scoring reference.
set -euo pipefail
DATA=${1:?CRV data root}
OUT=${2:?new output directory}
input=$(head -n 1 "$DATA/timecal/files_P4.txt")
name=$(basename "$input" .root)
python3 -m crvfastsim.objective.joint_data --inputs "$input" --out "$OUT" \
  --chunk 25000 --window-ns 400000 \
  --reference-accumulator "$DATA/timecal/emit_v2/$name.json"
