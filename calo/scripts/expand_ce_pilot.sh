#!/bin/bash
# Expand only; no event processing or compilation.
set -eo pipefail
BASE=${1:?base FCL}
POOL=${2:?worker-visible stop pool}
OUT=${3:?configuration output directory}
mkdir -p "$OUT"
cat "$BASE" > "$OUT/pilot.fcl"
printf '\nphysics.filters.TargetStopResampler.fileNames : [ "%s" ]\n' "$POOL" >> "$OUT/pilot.fcl"
fhicl-dump -c "$OUT/pilot.fcl" > "$OUT/pilot.expanded.fcl"
sha256sum "$OUT/pilot.expanded.fcl" > "$OUT/pilot.expanded.sha256"
