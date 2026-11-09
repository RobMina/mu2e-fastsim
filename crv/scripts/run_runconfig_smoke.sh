#!/bin/bash
# CPU smoke of runchain per named run configuration on its gen-test table (grid/crygen_local_test.sh), nominal
# conditions, tier1b_v7 library. usage: run_runconfig_smoke.sh <run_config> <gen.root> [extra runchain args]
# needs pyenv ana, PYTHONPATH (core + crv), CRVFS_DATA; output $CRVFS_DATA/runcfg_dev/smoke_<run_config>
RC=${1:?run config}; GEN=${2:?gen table}; shift 2
D=${CRVFS_DATA:?}; O=$D/runcfg_dev/smoke_$RC; mkdir -p "$O"
python3 -m crvfastsim runchain "$GEN" --run-config "$RC" --library "$D/tier1b_v7/lib_prod_as_20files.root" \
  --calib nominal --outdir "$O" "$@" > "$O/runchain.log" 2>&1
rc=$?
echo "$RC rc=$rc"; grep -E "Error|error|Traceback|timers:" "$O/runchain.log" | tail -5
[ $rc -eq 0 ] || tail -25 "$O/runchain.log"
