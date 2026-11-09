#!/bin/bash
# GPVM test of grid/fcl/ref_sim_<cfg>.fcl (G4 reference for run1a/run2): N events, timing and accepted count, then the
# CrvStep and gen dumps (CrvFastSim area). usage: ref_sim_local_test.sh <run1a|run2> [N=2000] [subrun=9101]
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
C=${1:?cfg}; N=${2:-2000}; SR=${3:-9101}
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); O=${CRVFS_DATA:?}/g4ref_test_$C; mkdir -p "$O" && cd "$O"
S=$(printf "001400_%08d" "$SR")
cat > shard.fcl <<EOF
#include "ref_sim_$C.fcl"
source.firstSubRun : $SR
services.SeedService.baseSeed : $((20268000 + SR))
outputs.PrimaryOutput.fileName : "dts.$C.$S.art"
outputs.GenOutput.fileName : "gen.$C.$S.art"
EOF
( muse setup SimJob MDC2025ay >/dev/null 2>&1; set -u; export FHICL_FILE_PATH=$H/fcl:$O:$FHICL_FILE_PATH MU2E_SEARCH_PATH=$H/fcl:$MU2E_SEARCH_PATH
  /usr/bin/time -f "sim %e s %M kB" mu2e -c shard.fcl -n "$N" > sim.log 2>&1; echo "sim rc=$?"; tail -1 sim.log
  grep -E "TrigReport Events total" sim.log | head -2 )
( cd "${CRVFS_CODE:?}/art" && muse setup >/dev/null 2>&1; set -u; cd "$O"
  mu2e -c CrvFastSim/fcl/dump_crvsteps.fcl -s "dts.$C.$S.art" -T "crvsteps.$S.root" > dump.log 2>&1; echo "dump rc=$?"
  mu2e -c CrvFastSim/fcl/dump_gen.fcl -s "gen.$C.$S.art" -T "gen.$S.root" > dumpgen.log 2>&1; echo "dumpgen rc=$?" )
