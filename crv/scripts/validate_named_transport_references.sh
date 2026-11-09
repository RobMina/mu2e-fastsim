#!/bin/bash
# Compare retained 20k and existing 50k references with frozen transport; never regenerate old candidates.
set -eo pipefail
D=${CRVFS_DATA:?set CRVFS_DATA}
REFS=${1:?completed named-reference dump directory}
OLD=${2:?existing 20k candidate ledger}
OUT=${3:?new output directory}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir "$OUT"
printf 'crv_head=%s\nPYTHONPATH=%s\nreferences=%s\nold_campaign=%s\nseed=1\n' \
  "$(git -C "$CODE" rev-parse HEAD)" "$PYTHONPATH" "$REFS" "$OLD" > "$OUT/manifest.txt"
sha256sum "$CODE/crvfastsim/transport.py" "$CODE/crvfastsim/analyses/transport_validation.py" \
  "$CODE/crvfastsim/analyses/v1compare.py" "$CODE/crvfastsim/cli.py" "$REFS/dump_manifest.txt" >> "$OUT/manifest.txt"
for cfg in run1a run2; do
  if [ "$cfg" = run1a ]; then RC=run1a_v01; PILOT=001400_00009111; ACCEPTED=8019; OLDACC=3326
  else RC=run2_v10; PILOT=001400_00009112; ACCEPTED=9593; OLDACC=3971; fi
  S=001400_00000101
  Q=$OUT/$cfg
  mkdir "$Q" "$Q/tables" "$Q/batches"
  /usr/bin/time -f 'transport_wall_s=%e maxrss_kb=%M' python3 -m crvfastsim transport \
    "$REFS/$cfg/gen.$S.root" --stage C --seed 1 --run-config "$RC" \
    --gdml "$D/geometry/mu2e_$RC.gdml" --outdir "$Q/tables" > "$Q/transport.log" 2>&1
  python3 -m crvfastsim transport-validation --gen "$REFS/$cfg/gen.$S.root" \
    --reference "$REFS/$cfg/crvsteps.$S.root" --candidate "$Q/tables/fastC.$S.root" \
    --config "$RC" --expected-events 50000 --expected-accepted "$ACCEPTED" \
    --gdml "$D/geometry/mu2e_$RC.gdml" --artifact "$REFS/$cfg/worker.$S.log" \
    --artifact "$REFS/$cfg/dump_crvsteps.$S.log" --artifact "$REFS/$cfg/dump_gen.$S.log" \
    --artifact "$Q/transport.log" --output "$Q/audit50k.json" > "$Q/audit50k.log" 2>&1
  python3 -m crvfastsim transport-validation --gen "$D/g4ref_test_$cfg/gen.$PILOT.root" \
    --reference "$D/g4ref_test_$cfg/crvsteps.$PILOT.root" --candidate "$OLD/${cfg}_tables/fastC.$PILOT.root" \
    --config "$RC" --expected-events 20000 --expected-accepted "$OLDACC" \
    --gdml "$D/geometry/mu2e_$RC.gdml" --artifact "$D/g4ref_test_$cfg/sim.log" \
    --artifact "$D/g4ref_test_$cfg/dump.log" --artifact "$D/g4ref_test_$cfg/dumpgen.log" \
    --artifact "$OLD/${cfg}_transport.log" --output "$Q/audit20k.json" > "$Q/audit20k.log" 2>&1
  python3 -m crvfastsim transport-validation --merge "$Q/audit20k.json" "$Q/audit50k.json" \
    --output "$Q/audit70k.json" > "$Q/audit70k.log" 2>&1
  python3 -m crvfastsim v1compare "$Q/tables/fastC.$S.root" --compare "$REFS/$cfg/crvsteps.$S.root" \
    --run-config "$RC" --emit "$Q/batches/50k.json" > "$Q/emit50k.log" 2>&1
  python3 -m crvfastsim v1compare "$OLD/${cfg}_tables/fastC.$PILOT.root" \
    --compare "$D/g4ref_test_$cfg/crvsteps.$PILOT.root" --run-config "$RC" \
    --emit "$Q/batches/20k.json" > "$Q/emit20k.log" 2>&1
  for size in 50k 70k; do
    if [ "$size" = 50k ]; then BATCH="$Q/batches/50k.json"; else BATCH="$Q/batches/*.json"; fi
    rc=0
    python3 -m crvfastsim v1compare "$BATCH" --merge --run-config "$RC" \
      --outdir "$Q/v1_$size" > "$Q/v1_$size.log" 2>&1 || rc=$?
    printf '%s %s exit=%s\n' "$cfg" "$size" "$rc" | tee -a "$OUT/exit_codes.txt"
    cat "$Q/v1_$size.log"
    if [ "$rc" -gt 1 ] || [ ! -f "$Q/v1_$size/v1compare_stats.json" ]; then exit 2; fi
  done
  cat "$Q/audit70k.log"
done
