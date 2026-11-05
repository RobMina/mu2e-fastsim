#!/bin/bash
# Partition explicit straight diagnostic chords in existing exports, without simulation.
set -eo pipefail
[[ $# -eq 2 ]] || { echo 'usage: tschord_audit.sh EXPORT_ROOT AUDIT_SUFFIX' >&2; exit 2; }
ROOT=$1
SUFFIX=$2
for ARM in run1a run1b; do
  python3 -m beamfastsim.tschord "$ROOT/geometry_${ARM}_v1/ts_geometry_${ARM}.gdml" \
    --root TS1Vacuum --root TS2Vacuum --root TS3Vacuum --root TS4Vacuum --root TS5Vacuum \
    --vacuum-material PSVacuum --vacuum-material DSVacuum --assume-straight --atol 0.000001 \
    --chord 3904 0 -3000 3904 0 -2850 --chord 2 0 0 -2 0 0 \
    --chord -3904 0 2925 -3904 0 2935 \
    --out "$ROOT/geometry_${ARM}_v1/chords_${SUFFIX}.json"
done
