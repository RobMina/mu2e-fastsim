#!/bin/bash
# Audit existing Run1A/Run2 candidates and retained references; no transport or G4 is rerun.
set -eo pipefail
D=${CRVFS_DATA:?set CRVFS_DATA}
OUT=${1:?candidate campaign directory}
TAG=${2:?new audit tag}
mkdir "$OUT/$TAG"
for cfg in run1a run2; do
  if [ "$cfg" = run1a ]; then RC=run1a_v01; S=001400_00009111; else RC=run2_v10; S=001400_00009112; fi
  python3 -m crvfastsim.analyses.transport_validation --gen "$D/g4ref_test_$cfg/gen.$S.root" \
    --reference "$D/g4ref_test_$cfg/crvsteps.$S.root" --baseline "$D/runcfg_dev/v1d_$RC/s1/fastC.$S.root" \
    --candidate "$OUT/${cfg}_tables/fastC.$S.root" --config "$RC" --expected-events 20000 \
    --gdml "$D/geometry/mu2e_$RC.gdml" --artifact "$D/g4ref_test_$cfg/shard.fcl" \
    --artifact "$D/g4ref_test_$cfg/sim.log" --artifact "$D/g4ref_test_$cfg/dump.log" \
    --artifact "$D/g4ref_test_$cfg/dumpgen.log" --artifact "$OUT/${cfg}_transport.log" \
    --output "$OUT/$TAG/$cfg.json" > "$OUT/$TAG/$cfg.log" 2>&1
  cat "$OUT/$TAG/$cfg.log"
done
