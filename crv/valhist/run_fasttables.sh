#!/bin/bash
# Fast tables for the table arms on g4ref_v1 shards, serially on one GPVM (numpy, CPU):
#   usage: run_fasttables.sh FIRST LAST STAGE INPUT OUTDIR [extra fasttables args]
#   INPUT = <dir>/<prefix> under $CRVFS_DATA; the input file is $CRVFS_DATA/<dir>/<prefix>.<subrun>.root
#   STAGE reco : stock CrvRecoTableDumper files, e.g. valhist/g4ref_v1/a0_dump/valhist (A3: fast reco on stock digis)
#   STAGE chain|digis : step tables, e.g. g4ref_v1_tables/crvsteps (A4/A2 on G4 steps), tier1b_v8/fast1b (E2E)
#   output $CRVFS_DATA/OUTDIR/<stage>.<subrun>.root
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana >/dev/null 2>&1 || true
set -u
A=${1:?first}; B=${2:?last}; STAGE=${3:?stage}; IN=${4:?input}; OD=${5:?outdir}; shift 5
D=${CRVFS_DATA:?source site.env}; O=$D/$OD; mkdir -p "$O/logs"
cd "$CRVFS_CODE"
for k in $(seq "$A" "$B"); do
  S=$(printf "001400_%08d" "$k")
  [ -s "$O/$STAGE.$S.root" ] && continue
  SRCF=$D/$IN.$S.root
  PYTHONPATH=$CRVFS_CODE:$M2FS_CODE/core python3 -m valhist.fasttables "$STAGE" "$SRCF" "$O/$STAGE.$S.root" \
    --calib "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv" --deviation "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" \
    --seed "$k" "$@" > "$O/logs/$STAGE.$S.log" 2>&1
  echo "$S rc=$? $(date +%T)"
done
