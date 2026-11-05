#!/bin/bash
# Compile selected existing GDML subtrees, without invoking simulation.
set -eo pipefail
[[ $# -eq 2 ]] || { echo 'usage: tsgeom_audit.sh EXPORT_ROOT AUDIT_SUFFIX' >&2; exit 2; }
ROOT=$1
SUFFIX=$2
for ARM in run1a run1b; do
  python3 -m beamfastsim.tsgeom "$ROOT/geometry_${ARM}_v1/ts_geometry_${ARM}.gdml" \
    --root TS1Vacuum --root TS2Vacuum --root TS3Vacuum --root TS4Vacuum --root TS5Vacuum \
    --vacuum-material PSVacuum --vacuum-material DSVacuum \
    --point 3904 0 -3504 --point 4079 0 -3504 --point 452 90 0 --point 452 -50 0 \
    --point 0 0 0 --point -3904 0 3554 --point -3754 0 3554 \
    --out "$ROOT/geometry_${ARM}_v1/clearance_${SUFFIX}.json"
done
