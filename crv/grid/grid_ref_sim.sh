#!/usr/bin/env bash
# Grid worker: one shard of the G4 reference sample (stock ExtractedCRY, Musing SimJob MDC2025ay from CVMFS).
# Shard k = PROCESS: subrun k+1, baseSeed BASESEED+k, NEVTS events. Writes dts (accepted) + gen (all primaries).
set -o pipefail
OUTDIR="${1:?need OUTDIR (pnfs scratch)}"
NEVTS="${NEVTS:?}"; BASESEED="${BASESEED:?}"; FCL_PNFS="${FCL_PNFS:?}"; MUSING="${MUSING:-MDC2025ay}"
K="${PROCESS:?}"; SUBRUN=$((K + 1)); SEED=$((BASESEED + K))
SEQ=$(printf "001400_%08d" "$SUBRUN")
OWNER="${OWNER:-mu2e}"
DTS="dts.${OWNER}.CosmicCRYExtractedRef.v0.${SEQ}.art"; GEN="gen.${OWNER}.CosmicCRYExtractedRef.v0.${SEQ}.art"
echo "=== grid_ref_sim: shard $K subrun $SUBRUN seed $SEED nevts $NEVTS host $(hostname) $(date -u)"

source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
muse setup SimJob "$MUSING" || true
setup ifdhc 2>/dev/null || true
which mu2e ifdh || { echo "ERROR: mu2e/ifdh missing"; exit 2; }

if ifdh ls "$OUTDIR/$DTS" >/dev/null 2>&1 && ifdh ls "$OUTDIR/$GEN" >/dev/null 2>&1; then
  echo "SKIP: outputs exist"; exit 0
fi

mkdir -p fcl && ifdh cp "$FCL_PNFS/ref_sim_extracted.fcl" fcl/ref_sim_extracted.fcl || { echo "ERROR: fcl"; exit 2; }
cat > fcl/shard.fcl <<EOF
#include "ref_sim_extracted.fcl"
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
