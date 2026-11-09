#!/bin/bash
# Add current paired diagnostics to saved named candidates without rerunning transport or G4.
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
  "$REFS/dump_manifest.txt" "$OLD/extracted20/manifest.txt" "$CAND/manifest.txt" >> "$OUT/manifest.txt"
for cfg in run1a run2; do
  if [ "$cfg" = run1a ]; then RC=run1a_v01; PILOT=001400_00009111; ACCEPTED=8019; OLDACC=3326
  else RC=run2_v10; PILOT=001400_00009112; ACCEPTED=9593; OLDACC=3971; fi
  S=001400_00000101
  Q=$OUT/$cfg
  mkdir "$Q"
  python3 -m crvfastsim transport-validation --gen "$REFS/$cfg/gen.$S.root" \
    --reference "$REFS/$cfg/crvsteps.$S.root" --candidate "$CAND/$cfg/tables/fastC.$S.root" \
    --config "$RC" --expected-events 50000 --expected-accepted "$ACCEPTED" \
    --gdml "$D/geometry/mu2e_$RC.gdml" --artifact "$REFS/$cfg/worker.$S.log" \
    --artifact "$REFS/$cfg/dump_crvsteps.$S.log" --artifact "$REFS/$cfg/dump_gen.$S.log" \
    --artifact "$CAND/$cfg/transport.log" --output "$Q/audit50k.json" > "$Q/audit50k.log" 2>&1
  python3 -m crvfastsim transport-validation --gen "$D/g4ref_test_$cfg/gen.$PILOT.root" \
    --reference "$D/g4ref_test_$cfg/crvsteps.$PILOT.root" --candidate "$OLD/${cfg}_tables/fastC.$PILOT.root" \
    --config "$RC" --expected-events 20000 --expected-accepted "$OLDACC" \
    --gdml "$D/geometry/mu2e_$RC.gdml" --artifact "$D/g4ref_test_$cfg/sim.log" \
    --artifact "$D/g4ref_test_$cfg/dump.log" --artifact "$D/g4ref_test_$cfg/dumpgen.log" \
    --artifact "$OLD/${cfg}_transport.log" --output "$Q/audit20k.json" > "$Q/audit20k.log" 2>&1
  python3 -m crvfastsim transport-validation --merge "$Q/audit20k.json" "$Q/audit50k.json" \
    --output "$Q/audit70k.json" > "$Q/audit70k.log" 2>&1
  python3 -m crvfastsim transport-validation --summary "$Q/audit70k.json" > "$Q/summary70k.log" 2>&1
  cat "$Q/summary70k.log"
done
