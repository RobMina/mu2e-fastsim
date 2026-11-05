#!/bin/bash
# Audit existing geometry exports without invoking simulation.
set -eo pipefail
[[ $# -eq 2 ]] || { echo 'usage: tsgdml_audit.sh EXPORT_ROOT AUDIT_SUFFIX' >&2; exit 2; }
ROOT=$1
SUFFIX=$2
for ARM in run1a run1b; do
  python3 -m beamfastsim.tsgdml "$ROOT/geometry_${ARM}_v1/ts_geometry_${ARM}.gdml" \
    --root TS1Vacuum --root TS2Vacuum --root TS3Vacuum --root TS4Vacuum --root TS5Vacuum \
    --out "$ROOT/geometry_${ARM}_v1/structure_${SUFFIX}.json"
done
