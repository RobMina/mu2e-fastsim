#!/bin/bash
# Grid worker: F-fast deadtime frames for ONE Run 1A CE+Mix1BB subrun (crvfastsim.veto.fastframes), plus the stock
# Run 1A response + reco re-run on the same dig-kept CrvSteps (redigi control).
#   grid_fastframes_run1a.sh OUTDIR
# PROCESS selects line PROCESS+1 of LIST_PNFS ("<subrun> <dig path> <mcs path> <nts path>"). Release: SimJob MDC2025ay
# from CVMFS (art + CrvFastSim dumper libs/fcl from DUMPER_PNFS); Python: Mu2e ana env (numpy path, no JAX); code: git
# archive of core + crv (CODE_PNFS); lookup tables and CRVPhoton table from CVMFS / CONDPHOTON_PNFS. Inputs streamed
# with xrootd. Outputs to OUTDIR: frames_fast_<s>.parquet, crvreco_mcs_<s>.root, crvredigi_<s>.root, logs.
# Skips a subrun whose frames output exists. No `set -e`; `set -u` after the setups.
OUTDIR="${1:?outdir}"
P="${PROCESS:-0}"
die() { echo "FATAL[fastframes $P]: $*" >&2; exit 1; }
echo "=== fastframes PROCESS=$P host=$(hostname) $(date -u)"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || echo "(setupmu2e-art rc=$?)"
muse setup SimJob MDC2025ay || echo "(muse setup rc=$?)"
command -v ifdh >/dev/null 2>&1 || setup ifdhc || die "no ifdh"
set -u
ifdh cp "${LIST_PNFS:?}" list.txt || die "stage list"
read -r S DIG MCS NTS <<< "$(sed -n "$((P + 1))p" list.txt)"
[ -n "${S:-}" ] || die "no line $((P + 1)) in list"
if ifdh ls "$OUTDIR/frames_fast_$S.parquet" >/dev/null 2>&1; then echo "SKIP: frames_fast_$S.parquet exists"; exit 0; fi
x() { echo "$1" | sed 's|^/pnfs/|root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/|'; }
ifdh cp "${DUMPER_PNFS:?}" dumper.tgz && mkdir -p dumper && tar -xzf dumper.tgz -C dumper || die "dumper tarball"
ifdh cp "${CODE_PNFS:?}" code.tgz && mkdir -p code && tar -xzf code.tgz -C code || die "code tarball"
ifdh cp "${CONDPHOTON_PNFS:?}" crvphoton.txt || die "CRVPhoton table"
cat code/VERSION
export CET_PLUGIN_PATH="$PWD/dumper/lib:$CET_PLUGIN_PATH" LD_LIBRARY_PATH="$PWD/dumper/lib:$LD_LIBRARY_PATH"
export FHICL_FILE_PATH="$PWD/dumper/fcl:$FHICL_FILE_PATH"
echo "--- subrun $S"
mu2e -c CrvFastSim/fcl/dump_crvsteps_mix.fcl -s "$(x "$DIG")" -T mixsteps.root > dump_mix_$S.log 2>&1 || { tail -20 dump_mix_$S.log; die "dump mix"; }
mu2e -c CrvFastSim/fcl/dump_crvreco_from_mcs_run1a.fcl -s "$(x "$MCS")" -T crvreco_mcs_$S.root > dump_mcs_$S.log 2>&1 || { tail -20 dump_mcs_$S.log; die "dump mcs"; }
mu2e -c CrvFastSim/fcl/redigi_crv_from_dig_run1a.fcl -s "$(x "$DIG")" -T crvredigi_$S.root > redigi_$S.log 2>&1 || { tail -20 redigi_$S.log; die "redigi"; }
grep -h "TrigReport Events total" dump_mix_$S.log dump_mcs_$S.log redigi_$S.log
ifdh cp "$NTS" nts.root || die "stage nts"
pyenv ana >/dev/null 2>&1 || echo "(pyenv ana rc=$?)"
L=$PWD/lookup; mkdir -p $L
for f in LookupTable_6000_0 photonMap.root; do cp /cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0/$f $L/ || die "lookup $f"; done
/usr/bin/time -v env CRVFS_LOOKUP_DIR=$L PYTHONPATH=$PWD/code/core:$PWD/code/crv \
  MU2E_SEARCH_PATH=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_33_00 \
  python3 -m crvfastsim.veto.fastframes mixsteps.root --conditions crvreco_mcs_$S.root --photon-dev crvphoton.txt \
  --pbtime nts.root --chunk "${CHUNK:-500}" --out frames_fast_$S.parquet > frames_$S.log 2>&1
rc=$?; tail -1 frames_$S.log | cut -c1-200; grep -E "Maximum resident|Elapsed" frames_$S.log
[ $rc -eq 0 ] && [ -s frames_fast_$S.parquet ] || { tail -20 frames_$S.log; die "fastframes rc=$rc"; }
rm -f mixsteps.root nts.root
for f in crvreco_mcs_$S.root crvredigi_$S.root frames_$S.log dump_mix_$S.log dump_mcs_$S.log redigi_$S.log; do
  ifdh cp $f "$OUTDIR/$f" || echo "(stage-out $f failed)"
done
ifdh cp frames_fast_$S.parquet "$OUTDIR/frames_fast_$S.parquet" || die "stage-out frames"
echo "=== done $S"
