#!/bin/bash
# Grid worker, stage 2 of F-fast from beam frames: EVENTS beam-only events mixed from the compact *Cat step tables
# (TABLES_PNFS: <Mixer>/steps.*.parquet from stage 1, copied in) and run through the fast Run 1A response + reco
# (crvfastsim.veto.mixframes); seed = SEED0 + PROCESS. Output frames_mix_<seed>.parquet + .json to OUTDIR.
OUTDIR="${1:?outdir}"; P="${PROCESS:-0}"; SEED=$(( ${SEED0:-1000} + P ))
die() { echo "FATAL[mixframes $SEED]: $*" >&2; exit 1; }
echo "=== mixframes seed=$SEED events=${EVENTS:?} host=$(hostname) $(date -u)"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || echo "(setupmu2e-art rc=$?)"
command -v ifdh >/dev/null 2>&1 || setup ifdhc || die "no ifdh"
set -u
OUT=frames_mix_$SEED.parquet
if ifdh ls "$OUTDIR/$OUT" >/dev/null 2>&1; then echo "SKIP: $OUT exists"; exit 0; fi
ifdh cp "${CODE_PNFS:?}" code.tgz && mkdir -p code && tar -xzf code.tgz -C code || die "code tarball"
ifdh cp "${COND_PNFS:?}" crvreco_mcs.root || die "conditions"
ifdh cp "${CONDPHOTON_PNFS:?}" crvphoton.txt || die "CRVPhoton table"
cat code/VERSION
# a random subset of each mixer's tables (seeded): all of them are ~18 GB / ~22 GB in memory; FILES_<Mixer> caps the
# count per job (default: every table). A job's frames are drawn from its subset, so jobs use different frames.
mkdir -p tables
for m in EleBeamFlashCat MuBeamFlashCat MuStopPileupCat NeutralsFlashCat; do
  mkdir -p tables/$m
  ifdh ls "${TABLES_PNFS:?}/$m" 2>/dev/null | grep "\.parquet$" > all_$m.txt
  cap=$(eval echo "\${FILES_$m:-0}")
  if [ "$cap" -gt 0 ]; then shuf -n "$cap" --random-source=<(openssl enc -aes-256-ctr -pass pass:"$SEED$m" -nosalt -pbkdf2 </dev/zero 2>/dev/null) all_$m.txt > use_$m.txt; else cp all_$m.txt use_$m.txt; fi
  while read -r f; do ifdh cp "$f" tables/$m/$(basename "$f") >/dev/null || die "table $f"; done < use_$m.txt
  echo "$m: $(ls tables/$m | wc -l) of $(wc -l < all_$m.txt) tables, $(du -sh tables/$m | cut -f1)"
done
L=$PWD/lookup; mkdir -p $L
for f in LookupTable_6000_0 photonMap.root; do cp /cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0/$f $L/ || die "lookup $f"; done
/usr/bin/time -v env CRVFS_LOOKUP_DIR=$L PYTHONPATH=$PWD/code/core:$PWD/code/crv \
  MU2E_SEARCH_PATH=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_33_00 \
  /cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python3 -m crvfastsim.veto.mixframes --tables tables --events "$EVENTS" \
  --seed "$SEED" --conditions crvreco_mcs.root --photon-dev crvphoton.txt --chunk "${CHUNK:-500}" --out "$OUT" > mix_$SEED.log 2>&1
rc=$?; tail -c 1500 mix_$SEED.log; grep -E "Maximum resident|Elapsed" mix_$SEED.log
[ $rc -eq 0 ] && [ -s "$OUT" ] || die "mixframes rc=$rc"
for f in ${OUT%.parquet}.json mix_$SEED.log; do ifdh cp $f "$OUTDIR/$f" || echo "(stage-out $f failed)"; done
ifdh cp "$OUT" "$OUTDIR/$OUT" || die "stage-out frames"
echo "=== done seed $SEED"
