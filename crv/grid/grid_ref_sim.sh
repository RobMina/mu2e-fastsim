#!/usr/bin/env bash
# Grid worker: one shard of the G4 reference sample (stock ExtractedCRY, Musing SimJob MDC2025ay from CVMFS).
# Shard k = PROCESS: subrun FIRST_SUBRUN+k (default 1), baseSeed BASESEED+FIRST_SUBRUN-1+k, NEVTS events. Writes dts
# (accepted) + gen (all primaries). TAG names the files (default v0, as g4ref_v1).
set -o pipefail
OUTDIR="${1:?need OUTDIR (pnfs scratch)}"
NEVTS="${NEVTS:?}"; BASESEED="${BASESEED:?}"; FCL_PNFS="${FCL_PNFS:?}"; MUSING="${MUSING:-MDC2025ay}"
FIRST_SUBRUN="${FIRST_SUBRUN:-1}"; TAG="${TAG:-v0}"; CFG="${CFG:-extracted}"
K="${PROCESS:?}"; SUBRUN=$((FIRST_SUBRUN + K)); SEED=$((BASESEED + FIRST_SUBRUN - 1 + K))
SEQ=$(printf "001400_%08d" "$SUBRUN")
OWNER="${OWNER:-mu2e}"
DESC=CosmicCRYExtractedRef; [ "$CFG" = extracted ] || DESC=CosmicCRY${CFG}Ref
DTS="dts.${OWNER}.${DESC}.${TAG}.${SEQ}.art"; GEN="gen.${OWNER}.${DESC}.${TAG}.${SEQ}.art"
echo "=== grid_ref_sim: shard $K subrun $SUBRUN seed $SEED nevts $NEVTS host $(hostname) $(date -u)"

source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
muse setup SimJob "$MUSING" || true
setup ifdhc 2>/dev/null || true
which mu2e ifdh || { echo "ERROR: mu2e/ifdh missing"; exit 2; }

if ifdh ls "$OUTDIR/$DTS" >/dev/null 2>&1 && ifdh ls "$OUTDIR/$GEN" >/dev/null 2>&1; then
  echo "SKIP: outputs exist"; exit 0
fi

mkdir -p fcl && ifdh cp "$FCL_PNFS/ref_sim_extracted.fcl" fcl/ref_sim_extracted.fcl || { echo "ERROR: fcl"; exit 2; }
[ "$CFG" = extracted ] || ifdh cp "$FCL_PNFS/ref_sim_$CFG.fcl" fcl/ref_sim_$CFG.fcl || { echo "ERROR: fcl $CFG"; exit 2; }
[ "$CFG" = run2 ] && { ifdh cp "$FCL_PNFS/geom_cosmic_run2_v10.txt" fcl/geom_cosmic_run2_v10.txt || exit 2; export MU2E_SEARCH_PATH="$PWD/fcl:$MU2E_SEARCH_PATH"; }
cat > fcl/shard.fcl <<EOF
#include "ref_sim_$CFG.fcl"
source.firstSubRun : $SUBRUN
services.SeedService.baseSeed : $SEED
outputs.PrimaryOutput.fileName : "$DTS"
outputs.GenOutput.fileName : "$GEN"
EOF
export FHICL_FILE_PATH="$PWD/fcl:$FHICL_FILE_PATH"

/usr/bin/time -v mu2e -c fcl/shard.fcl -n "$NEVTS" > mu2e.log 2>&1
rc=$?
grep -E "TrigReport Events total|VmHWM|Elapsed|User time" mu2e.log
[ $rc -eq 0 ] || { tail -40 mu2e.log; echo "MU2E FAILED rc=$rc"; exit 3; }

ifdh mkdir_p "$OUTDIR/logs"
ifdh cp "$DTS" "$OUTDIR/$DTS" || { echo "COPY dts FAILED"; exit 5; }
ifdh cp "$GEN" "$OUTDIR/$GEN" || { echo "COPY gen FAILED"; exit 5; }
gzip -c mu2e.log > "mu2e.${SEQ}.log.gz" && ifdh cp "mu2e.${SEQ}.log.gz" "$OUTDIR/logs/mu2e.${SEQ}.log.gz"
echo "=== grid_ref_sim: done shard $K"
