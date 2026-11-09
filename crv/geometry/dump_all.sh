#!/bin/bash
# GDML world export + bar printout for each named run configuration, serially on one GPVM (geometry/README.md).
# usage: dump_all.sh [config ...]   (default run1a_v01 run2_v10; Run 1B uses the Run 1A world); outputs in $CRVFS_DATA/geometry
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup SimJob MDC2025ay >/dev/null 2>&1 || true
set -u
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); O=${CRVFS_DATA:?source site.env}/geometry
cd "$O" && export FHICL_FILE_PATH=$G/fcl:$FHICL_FILE_PATH MU2E_SEARCH_PATH=$G/fcl:$MU2E_SEARCH_PATH
for c in "${@:-run1a_v01 run2_v10}"; do
  for n in $c; do
    mu2e -c dump_geometry_$n.fcl -n 1 > dump_$n.log 2>&1
    echo "$n rc=$? $(ls -la mu2e_$n.gdml 2>/dev/null | awk '{print $5}') bytes $(grep -c '^CRSScintillatorBar_' dump_$n.log) bars $(date +%T)"
  done
done
