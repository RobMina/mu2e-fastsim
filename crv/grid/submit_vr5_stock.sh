#!/usr/bin/env bash
# submit_vr5_stock.sh NAME CODE [--first-line 31] [--nfiles 5] [--points FILE] [--nevts N] [--lifetime 2h] [--test] [--dry-run]
#
# V-R5 stock arm: every point of vr5_points.txt on NFILES dts files (filelist lines FIRST..FIRST+NFILES-1; default the
# five MDC2025as files 990-1122 whose CrvStep tables the fast arm replays), one grid job per (point, file), through the
# crvonly_v05 chain with the point's FHiCL overrides, ending in the crvfastsim.objective accumulator.
#   CODE  RCDS dir (or dropbox://Code.tar.bz2) of the crv_febphase muse tarball
#   Env   FILELIST (default /exp/mu2e/data/users/ram2aq/crvonly_mc/dts_filelist_as.txt), MCR (the analysis repo with fcl/)
# Sizing: crvonly_v05 production (cluster 30732054) 20-28 min per 25 k-event file, 1.2 GB; outputs under
# /pnfs/mu2e/scratch/users/$USER/vr5/NAME/{emit,nts,logs,config}. Merge per point:
#   python -m crvfastsim.objective "<emit>/<point>/*.json" --merge --compare <reference.json> --outdir <dir>
set -euo pipefail
NAME="${1:?name}"; CODE="${2:?RCDS dir or dropbox tarball}"; shift 2
FIRST=31; NF=5; NEVTS=-1; LIFE=2h; TEST=false; DRY=false
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; POINTS="$HERE/vr5_points.txt"
while [[ $# -gt 0 ]]; do case "$1" in
  --first-line) FIRST=$2; shift 2 ;; --nfiles) NF=$2; shift 2 ;; --points) POINTS=$2; shift 2 ;;
  --nevts) NEVTS=$2; shift 2 ;; --lifetime) LIFE=$2; shift 2 ;; --test) TEST=true; shift ;; --dry-run) DRY=true; shift ;;
  *) echo "Unknown option: $1" >&2; exit 1 ;; esac; done
FILELIST="${FILELIST:-/exp/mu2e/data/users/ram2aq/crvonly_mc/dts_filelist_as.txt}"
MCR="${MCR:-/exp/mu2e/app/users/ram2aq/analysis/mu2e_track_crv_mcresponse}"
OUTBASE="/pnfs/mu2e/scratch/users/$USER/vr5/$NAME"
WRAPPER="$HERE/grid_vr5_stock.sh"

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/vr5_cfg_XXXX")"; trap 'rm -rf "$STAGE" "$STAGE.cfg.tar.bz2"' EXIT
mkdir -p "$STAGE/fcl" "$STAGE/crvfastsim"
for f in digi_crvonly_extracted_v04.fcl digi_crvonly_extracted_v05.fcl reco_crvonly_extracted_v04.fcl \
         reco_crvonly_extracted_v05.fcl reco_kkts_extrap_nofilter_v04.fcl reco_smoke_kl_ts.fcl reco_evd_v03.fcl \
         ntuple_crvonly_v05.fcl ntuple_kkts_extrap_v04.fcl; do
  cp "$MCR/fcl/$f" "$STAGE/fcl/" || { echo "ERROR: missing fcl $f" >&2; exit 1; }
done
: > "$STAGE/points.txt"
grep -v '^#' "$POINTS" | while read -r pt kv; do
  [ -n "$pt" ] || continue
  echo "$pt" >> "$STAGE/points.txt"
  { echo '#include "digi_crvonly_extracted_v05.fcl"'
    for x in $kv; do echo "${x%%=*} : ${x#*=}"; done
    echo "outputs.Output.fileName : \"dig.owner.VR5$pt.version.sequencer.art\""; } > "$STAGE/fcl/digi_$pt.fcl"
done
LC_ALL=C grep -l '[^ -~]' "$STAGE"/fcl/*.fcl && { echo "ERROR: non-ASCII character in a staged fcl" >&2; exit 1; }
sed -n "${FIRST},$((FIRST + NF - 1))p" "$FILELIST" > "$STAGE/filelist.txt"
[ "$(wc -l < "$STAGE/filelist.txt")" -eq "$NF" ] || { echo "ERROR: filelist too short" >&2; exit 1; }
touch "$STAGE/crvfastsim/__init__.py"
tar -C "$HERE/../crvfastsim" --exclude='__pycache__' --exclude='*.pyc' -cf - objective | tar -C "$STAGE/crvfastsim" -xf -
tar -C "$STAGE" -cjf "$STAGE.cfg.tar.bz2" fcl filelist.txt points.txt crvfastsim
[ -n "${KEEP_STAGE:-}" ] && { rm -rf "$KEEP_STAGE"; cp -r "$STAGE" "$KEEP_STAGE"; }   # keep the staged config (fhicl-dump check)
NP=$(wc -l < "$STAGE/points.txt"); NJOBS=$((NP * NF)); $TEST && NJOBS=1

cat <<EOF
name     : $NAME   points $NP ($(tr '\n' ' ' < "$STAGE/points.txt"))
files    : $(sed -E 's|.*_([0-9]+)\.art|\1|' "$STAGE/filelist.txt" | tr '\n' ' ')(lines $FIRST-$((FIRST + NF - 1)))
jobs     : $NJOBS   nevts $NEVTS   lifetime $LIFE
code     : $CODE
config   : $(du -h "$STAGE.cfg.tar.bz2" | cut -f1) -> $OUTBASE/config/config.tar.bz2
outputs  : $OUTBASE/{emit,nts,logs}/
crv      : $(git -C "$HERE/.." rev-parse --short HEAD) $(git -C "$HERE/.." status --porcelain -- crvfastsim/objective | wc -l) uncommitted objective files
EOF
$DRY && { echo "Dry run only."; exit 0; }

set +u
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup ops >/dev/null 2>&1 || true
command -v ifdh >/dev/null 2>&1 || setup ifdhc
set -u
ifdh mkdir_p "$OUTBASE/config"
ifdh rm "$OUTBASE/config/config.tar.bz2" >/dev/null 2>&1 || true
ifdh cp "$STAGE.cfg.tar.bz2" "$OUTBASE/config/config.tar.bz2"
jobsub_submit -N "$NJOBS" -G mu2e --role=Analysis --expected-lifetime="$LIFE" \
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --lines='+FERMIHTC_AutoRelease=True' \
  --memory=3000MB --disk=6GB -e NFILES="$NF" -e NEVTS="$NEVTS" \
  --need-storage-modify="${OUTBASE#/pnfs}" --tar-file-name "$CODE" \
  "file://$WRAPPER" "$OUTBASE" 2>&1 | tee submit_out.txt
grep -qE "submitted to cluster" submit_out.txt || { echo "ERROR: no cluster submitted" >&2; rm -f submit_out.txt; exit 1; }
grep -E "submitted to cluster|job id" submit_out.txt; rm -f submit_out.txt
