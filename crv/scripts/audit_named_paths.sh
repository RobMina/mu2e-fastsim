#!/bin/bash
# Diagnose path geometry on existing named candidate/reference tables without changing transport.
set -eo pipefail
D=${CRVFS_DATA:?set CRVFS_DATA}
REFS=${1:?completed named-reference dump directory}
OLD=${2:?existing 20k candidate ledger}
CAND=${3:?existing 50k candidate campaign}
OUT=${4:?new output directory}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir "$OUT"
printf 'crv_head=%s\nPYTHONPATH=%s\nreferences=%s\nold_campaign=%s\nnew_campaign=%s\n' \
  "$(git -C "$CODE" rev-parse HEAD)" "$PYTHONPATH" "$REFS" "$OLD" "$CAND" > "$OUT/manifest.txt"
sha256sum "$CODE/crvfastsim/transport.py" "$CODE/crvfastsim/analyses/transport_validation.py" \
  "$CODE/crvfastsim/analyses/transport_paths.py" "$REFS/dump_manifest.txt" "$CAND/manifest.txt" >> "$OUT/manifest.txt"
for cfg in run1a run2; do
  if [ "$cfg" = run1a ]; then RC=run1a_v01; PILOT=001400_00009111
  else RC=run2_v10; PILOT=001400_00009112; fi
  S=001400_00000101
  Q=$OUT/$cfg
  mkdir "$Q"
  python3 -m crvfastsim transport-paths --gen "$REFS/$cfg/gen.$S.root" \
    --reference "$REFS/$cfg/crvsteps.$S.root" --candidate "$CAND/$cfg/tables/fastC.$S.root" \
    --config "$RC" --expected-events 50000 --gdml "$D/geometry/mu2e_$RC.gdml" \
    --output "$Q/paths50k.json" > "$Q/paths50k.log" 2>&1
  echo "$cfg 50k complete"
  python3 -m crvfastsim transport-paths --gen "$D/g4ref_test_$cfg/gen.$PILOT.root" \
    --reference "$D/g4ref_test_$cfg/crvsteps.$PILOT.root" --candidate "$OLD/${cfg}_tables/fastC.$PILOT.root" \
    --config "$RC" --expected-events 20000 --gdml "$D/geometry/mu2e_$RC.gdml" \
    --output "$Q/paths20k.json" > "$Q/paths20k.log" 2>&1
  echo "$cfg 20k complete"
  python3 -m crvfastsim transport-paths --merge "$Q/paths20k.json" "$Q/paths50k.json" \
    --output "$Q/paths70k.json" > "$Q/paths70k.log" 2>&1
  echo "$cfg 70k merge complete"
done
