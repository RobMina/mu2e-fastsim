#!/bin/bash
# Dump production dts.mu2e.CosmicCRYExtracted.MDC2025as files to flat tables with CrvStepTableDumper (tier-1b library).
# usage: dump_prod_as.sh FILELIST FIRST LAST   (lines FIRST..LAST of FILELIST, one dCache dts path per line); needs site.env
# One GPVM, serially (~15 s per 25 k-event file); output $CRVFS_DATA/prod_as_tables/crvsteps.<seq>.root (~100 MB each).
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${CRVFS_CODE:?source site.env}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$CRVFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
L=${1:?filelist}; A=${2:?first}; B=${3:?last}
O=$CRVFS_DATA/prod_as_tables
mkdir -p "$O/logs" && cd "$O"
sed -n "${A},${B}p" "$L" | while read -r F; do
  S=$(basename "$F" .art | sed 's/.*MDC2025as\.//')
  [ -s "crvsteps.$S.root" ] || mu2e -c CrvFastSim/fcl/dump_crvsteps.fcl -s "$F" -T "crvsteps.$S.root" > "logs/dump.$S.log" 2>&1
  echo "$S rc=$? $(date +%T)"
done
echo "done $(date -Is)"
