#!/bin/bash
# GPVM test of the gen-only FCLs (grid/fcl/gen_only_<cfg>.fcl): N events, then the gen table dump (CrvFastSim area).
# usage: crygen_local_test.sh <cfg: extracted|run1a|run2> [N=20000] [subrun=9001]; outputs in $CRVFS_DATA/crygen_test_<cfg>
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
C=${1:?cfg}; N=${2:-20000}; SR=${3:-9001}
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); O=${CRVFS_DATA:?}/crygen_test_$C; mkdir -p "$O" && cd "$O"
cat > shard.fcl <<EOF
#include "gen_only_$C.fcl"
source.firstSubRun : $SR
services.SeedService.baseSeed : $((20269000 + SR))
outputs.GenOutput.fileName : "gen.$C.$SR.art"
EOF
( muse setup SimJob MDC2025ay >/dev/null 2>&1; set -u; export FHICL_FILE_PATH=$H/fcl:$O:$FHICL_FILE_PATH MU2E_SEARCH_PATH=$H/fcl:$MU2E_SEARCH_PATH
  /usr/bin/time -f "gen %e s %M kB" mu2e -c shard.fcl -n "$N" > gen.log 2>&1; echo "gen rc=$?"; tail -1 gen.log )
( cd "${CRVFS_CODE:?}/art" && muse setup >/dev/null 2>&1; set -u; cd "$O"
  mu2e -c CrvFastSim/fcl/dump_gen.fcl -s "gen.$C.$SR.art" -T "gen.001400_$(printf %08d $SR).root" > dump.log 2>&1; echo "dump rc=$?" )
