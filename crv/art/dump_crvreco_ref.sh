#!/bin/bash
# dump_crvreco_ref.sh <mc|mcnonoise|data> [N] -- make the CRV reco reference tables for crvfastsim.reco.
#   mc   : one dts.mu2e.CosmicCRYExtracted.MDC2025as file -> stock digi + CRV reco -> crvreco_mc.<seq>.root
#   mcnonoise : the same with CrvWaveforms.noise 0 -> crvreco_mcnonoise.<seq>.root (reference for vr1)
#   data : one raw.mu2e.trk.vst.124155 file -> unpack (dig kept in /tmp) -> CRV reco -> crvreco_data.<tag>.root
# N events (default 20000 mc, 47000 data). Outputs in $CRVFS_DATA/reco_ref/. One job per interactive machine.
# Needs site.env: CRVFS_CODE, CRVFS_DATA; for data also CRVFS_FCL_PATH (fcl dirs holding unpack_data_vst_fullmap_head.fcl)
# and CRVFS_SEARCH_PATH (root its payloads resolve under).
# No `set -u` before the setup (setupmu2e-art.sh).
WHAT=${1:?usage: dump_crvreco_ref.sh <mc|mcnonoise|data> [N]}
AREA="${CRVFS_CODE:?source site.env}/art"
OUT="${CRVFS_DATA:?source site.env}/reco_ref"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "$AREA" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
export FHICL_FILE_PATH="${CRVFS_FCL_PATH:-}:$FHICL_FILE_PATH"
[ -n "${CRVFS_SEARCH_PATH:-}" ] && export MU2E_SEARCH_PATH="$CRVFS_SEARCH_PATH:$MU2E_SEARCH_PATH"
mkdir -p "$OUT/logs" && cd "$OUT"
case "$WHAT" in
  mc|mcnonoise)
    N=${2:-20000}
    F=${DTS_FILE:?set DTS_FILE to a dts.mu2e.CosmicCRYExtracted.MDC2025as file path}
    SEQ=$(basename "$F" .art | sed 's/.*\.//')
    FCL=CrvFastSim/fcl/dump_crvreco_from_dts.fcl; [ "$WHAT" = mcnonoise ] && FCL=CrvFastSim/fcl/dump_crvreco_from_dts_nonoise.fcl
    /usr/bin/time -f "%e s wall %U s user %M kB" mu2e -c "$FCL" -s "$F" -n "$N" \
      -T "crvreco_$WHAT.$SEQ.root" > "logs/$WHAT.$SEQ.log" 2> "logs/$WHAT.$SEQ.time"
    echo "$WHAT $SEQ exit=$? $(tail -1 logs/$WHAT.$SEQ.time)" ;;
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
