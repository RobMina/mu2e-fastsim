#!/bin/bash
# Evaluate uniform-field diagnostic arcs in existing exports, not production tracks.
set -eo pipefail
[[ $# -eq 2 ]] || { echo 'usage: tsarc_audit.sh EXPORT_ROOT AUDIT_SUFFIX' >&2; exit 2; }
ROOT=$1
SUFFIX=$2
for ARM in run1a run1b; do
  python3 -m beamfastsim.tsarc "$ROOT/geometry_${ARM}_v1/ts_geometry_${ARM}.gdml" \
    --root TS1Vacuum --root TS2Vacuum --root TS3Vacuum --root TS4Vacuum --root TS5Vacuum \
    --vacuum-material PSVacuum --vacuum-material DSVacuum --field 0 0 1 \
    --position-error-mm 0 --geometry-error-mm 0 --atol 0.000001 --vacuum-prefix \
    --arc 3904 0 -3000 0 0 100 13 150 \
    --arc 2 0 0 -100 0 0 13 4 \
    --arc -3904 0 2925 0 0 100 13 10 \
    --out "$ROOT/geometry_${ARM}_v1/arcs_${SUFFIX}.json"
done
