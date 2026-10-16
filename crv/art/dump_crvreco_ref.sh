#!/bin/bash
# dump_crvreco_ref.sh <mc|data> [N] -- make the CRV reco reference tables for crvfastsim.reco with CrvRecoTableDumper.
#   mc   : one dts.mu2e.CosmicCRYExtracted.MDC2025as file -> stock digi + CRV reco -> crvreco_mc.<seq>.root
#   data : one raw.mu2e.trk.vst.124155 file -> unpack (dig kept in /tmp) -> CRV reco -> crvreco_data.<tag>.root
# N events (default 20000 mc, 47000 data). Outputs in $CRVFS_DATA/reco_ref/. One job per interactive machine.
# Needs site.env (CRVFS_CODE, CRVFS_DATA) and CRVFS_FCL_PATH = the fcl dirs holding unpack_data_vst_fullmap_head.fcl.
# No `set -u` before the setup (setupmu2e-art.sh).
WHAT=${1:?usage: dump_crvreco_ref.sh <mc|data> [N]}
AREA="${CRVFS_CODE:?source site.env}/art"
OUT="${CRVFS_DATA:?source site.env}/reco_ref"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "$AREA" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
export FHICL_FILE_PATH="${CRVFS_FCL_PATH:-}:$FHICL_FILE_PATH"
mkdir -p "$OUT/logs" && cd "$OUT"
case "$WHAT" in
  mc)
    N=${2:-20000}
    F=${DTS_FILE:?set DTS_FILE to a dts.mu2e.CosmicCRYExtracted.MDC2025as file path}
    SEQ=$(basename "$F" .art | sed 's/.*\.//')
    /usr/bin/time -f "%e s wall %U s user %M kB" mu2e -c CrvFastSim/fcl/dump_crvreco_from_dts.fcl -s "$F" -n "$N" \
      -T "crvreco_mc.$SEQ.root" > "logs/mc.$SEQ.log" 2> "logs/mc.$SEQ.time"
    echo "mc $SEQ exit=$? $(tail -1 logs/mc.$SEQ.time)" ;;
  data)
    N=${2:-47000}
    F=${RAW_FILE:?set RAW_FILE to a raw.mu2e.trk.vst file path}
    TAG=$(basename "$F" .art | sed 's/^raw\.mu2e\.trk\.vst\.//')
    TMP=/tmp/$USER/crvreco_ref && mkdir -p "$TMP"
    mu2e -c unpack_data_vst_fullmap_head.fcl -s "$F" -n "$N" -o "$TMP/dig.$TAG.art" > "logs/unpack.$TAG.log" 2>&1
    rc=$?; echo "unpack $TAG exit=$rc"; [ $rc -eq 0 ] || exit 1
    /usr/bin/time -f "%e s wall %U s user %M kB" mu2e -c CrvFastSim/fcl/dump_crvreco_from_dig.fcl -s "$TMP/dig.$TAG.art" \
      -T "crvreco_data.$TAG.root" > "logs/data.$TAG.log" 2> "logs/data.$TAG.time"
    rc=$?; rm -f "$TMP/dig.$TAG.art"
    echo "data $TAG exit=$rc $(tail -1 logs/data.$TAG.time)" ;;
  *) echo "unknown $WHAT"; exit 2 ;;
esac
echo DUMP_CRVRECO_REF_DONE
