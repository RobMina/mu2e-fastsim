#!/bin/bash
# grid_vr5_stock.sh OUTBASE -- grid job: one (point, dts file) of the V-R5 stock arm. The crvonly_v05 chain (digi with the
# point's FHiCL overrides -> reco_crvonly_extracted_v05 -> ntuple_crvonly_v05), then the crvfastsim.objective accumulator
# of the ntuple. Submitted by submit_vr5_stock.sh.
#
# Code   : the muse tarball of the crv_febphase area ($INPUT_TAR_DIR_LOCAL/Code)
# Config : OUTBASE/config/config.tar.bz2 -- fcl/ (digi_<point>.fcl and the analysis-repo fcls), filelist.txt (dts paths),
#          points.txt (point names in PROCESS order), crvfastsim/ (the objective package)
# Output : OUTBASE/nts/<point>/nts.<point>.<seq>.root, OUTBASE/emit/<point>/<seq>.json, OUTBASE/logs/<point>.<seq>.time.txt
# Env    : NFILES (dts files per point), NEVTS (default -1). PROCESS -> point PROCESS / NFILES, file PROCESS % NFILES.
#
# No set -e / set -u before the mu2e setup (both kill the job silently). Every failure names itself.
OUTBASE="${1:?need OUTBASE on /pnfs}"
NFILES="${NFILES:?}"; NEVTS="${NEVTS:--1}"; P="${PROCESS:-0}"
die() { echo "FATAL[vr5 PROCESS $P]: $*" >&2; exit 1; }
echo "=== grid_vr5_stock: PROCESS=$P NFILES=$NFILES NEVTS=$NEVTS host=$(hostname) $(date -u) ==="

echo "--- step 1: environment ---"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || echo "  (setupmu2e-art.sh rc=$?)"
setup muse || echo "  (setup muse rc=$?)"
CODE="${INPUT_TAR_DIR_LOCAL:-$PWD}/Code"
[ -f "$CODE/setup.sh" ] || { ls -la "${INPUT_TAR_DIR_LOCAL:-$PWD}" >&2; die "no Code/setup.sh"; }
source "$CODE/setup.sh" || echo "  (Code/setup.sh rc=$?)"
command -v ifdh >/dev/null 2>&1 || setup ifdhc || die "no ifdh"
command -v mu2e >/dev/null 2>&1 || die "mu2e not on PATH after Code/setup.sh"
PY=/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python
[ -x "$PY" ] || die "no interpreter at $PY"

echo "--- step 2: config ---"
ifdh cp "$OUTBASE/config/config.tar.bz2" ./config.tar.bz2 || die "stage config tarball"
mkdir -p cfg && tar -C cfg -xjf config.tar.bz2 || die "untar config"
export FHICL_FILE_PATH="$PWD/cfg/fcl:$FHICL_FILE_PATH"
[ -d cfg/textfiles ] && export MU2E_SEARCH_PATH="$PWD/cfg/textfiles:$MU2E_SEARCH_PATH"   # DbService text tables (digi only)
PT=$(sed -n "$((P / NFILES + 1))p" cfg/points.txt); [ -n "$PT" ] || die "no point for PROCESS $P"
IN=$(sed -n "$((P % NFILES + 1))p" cfg/filelist.txt); [ -n "$IN" ] || die "no dts for PROCESS $P"
B=$(basename "$IN" .art); SEQ="${B##*.}"
N="nts.$PT.$SEQ.root"
echo "  point $PT  input $IN (seq $SEQ)"
PYTHONPATH="$PWD/cfg" "$PY" -c "import crvfastsim.objective.crvresponse" || die "objective import failed"
if ifdh ls "$OUTBASE/emit/$PT/$SEQ.json" 2>/dev/null | grep -q "$SEQ.json"; then echo "  SKIP: emit exists"; exit 0; fi

echo "--- step 3: stage input ---"
ifdh cp "$IN" ./input.art || die "stage input $IN"

stage() {  # stage <name> <fcl> <in> <outflag> <out>
  /usr/bin/time -f "$1 %e s wall %U s user %M kB" -a -o times.txt \
    mu2e -c "$2" -s "$3" -n "$NEVTS" "$4" "$5" > "$1.log" 2>&1
  local rc=$?
  grep -E "TrigReport Events total" "$1.log" | head -1
  [ $rc -eq 0 ] && [ -s "$5" ] || { tail -40 "$1.log" >&2; die "$1 failed rc=$rc"; }
}
echo "--- step 4: digi ($PT) ---"; stage digi "digi_$PT.fcl" input.art -o dig.art; rm -f input.art
grep -E "photonYieldScaleFactor|CrossTalkProb|timeConstant|ThermalRate|textFile" "cfg/fcl/digi_$PT.fcl" || true
echo "--- step 5: reco ---";       stage reco reco_crvonly_extracted_v05.fcl dig.art -o mcs.art; rm -f dig.art
echo "--- step 6: ntuple ---";     stage ntuple ntuple_crvonly_v05.fcl mcs.art -T "$N"; rm -f mcs.art
echo "--- step 7: objective accumulator ---"
/usr/bin/time -f "emit %e s wall %M kB" -a -o times.txt env MPLCONFIGDIR="$PWD/mpl" PYTHONPATH="$PWD/cfg" \
  "$PY" -m crvfastsim.objective "$N" --emit "$SEQ.json" > emit.log 2>&1 || { tail -20 emit.log >&2; die "emit failed"; }
cat times.txt

echo "--- step 8: copy out ---"
ifdh mkdir_p "$OUTBASE/emit/$PT" && ifdh cp "$SEQ.json" "$OUTBASE/emit/$PT/$SEQ.json" || die "copy emit"
ifdh mkdir_p "$OUTBASE/nts/$PT" && ifdh cp "$N" "$OUTBASE/nts/$PT/$N" || echo "  (ntuple copy failed)"
ifdh mkdir_p "$OUTBASE/logs" && ifdh cp times.txt "$OUTBASE/logs/$PT.$SEQ.time.txt" || echo "  (times copy failed)"
echo "=== done PROCESS $P point $PT seq $SEQ $(date -u) ==="
