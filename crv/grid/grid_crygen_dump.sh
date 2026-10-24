#!/usr/bin/env bash
# Grid worker: dump one crygen_v1 gen art file to a flat gen table (CrvStepTableDumper, dump_gen.fcl).
# Shard k = PROCESS: subrun SUBRUN0+k. The dumper library + fcl come as a small tarball (DUMPER_PNFS); the release is
# SimJob MDC2025ay from CVMFS. Input streamed from pnfs (xrootd), output table copied to OUTDIR. Skips existing outputs.
set -o pipefail
OUTDIR="${1:?need OUTDIR}"; INDIR="${INDIR:?}"; SUBRUN0="${SUBRUN0:?}"; DUMPER_PNFS="${DUMPER_PNFS:?}"; OWNER="${OWNER:?}"
K="${PROCESS:?}"; S=$(printf "001400_%08d" $((SUBRUN0 + K)))
CFG="${CFG:-extracted}"; DESC=CosmicCRYExtractedGen; [ "$CFG" = extracted ] || DESC=CosmicCRY${CFG}Gen
IN="gen.${OWNER}.${DESC}.v1.${S}.art"; OUT="gen.${S}.root"
echo "=== grid_crygen_dump: $S host $(hostname) $(date -u)"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
muse setup SimJob MDC2025ay || true
setup ifdhc 2>/dev/null || true
if ifdh ls "$OUTDIR/$OUT" >/dev/null 2>&1; then echo "SKIP: $OUT exists"; exit 0; fi
ifdh cp "$DUMPER_PNFS" dumper.tgz && mkdir -p dumper && tar -xzf dumper.tgz -C dumper || { echo "ERROR: dumper"; exit 2; }
export CET_PLUGIN_PATH="$PWD/dumper/lib:$CET_PLUGIN_PATH" LD_LIBRARY_PATH="$PWD/dumper/lib:$LD_LIBRARY_PATH"
export FHICL_FILE_PATH="$PWD/dumper/fcl:$FHICL_FILE_PATH"
URL=$(echo "$INDIR/$IN" | sed 's|^/pnfs/|root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/|')
mu2e -c CrvFastSim/fcl/dump_gen.fcl -s "$URL" -T "$OUT" > dump.log 2>&1 || { tail -30 dump.log; echo "DUMP FAILED"; exit 3; }
grep -E "TimeReport CPU|Events total" dump.log
ifdh cp "$OUT" "$OUTDIR/$OUT" || { echo "COPY FAILED"; exit 5; }
echo "=== done $S"
