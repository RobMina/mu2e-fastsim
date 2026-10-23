#!/bin/bash
# Offline PR crv-fiber-speed, default identity: digi_nominal.fcl of fiberspeed_ab in the unpatched crv_febphase area
# (febPhaseMode unset = perFEB = stock), then reco in the PR area and ntuple in crv_febphase exactly as the A/B arms.
# Production FHiCL comes from crv_febphase and its build (the PR area has only Offline). Compare nts.unpatched.root with
# nts.nominal.root (crvextrap abidentical --rtol 0). One GPVM, ~3 min for 4000 events. The crv_fiberspeed area was
# removed 2026-10-23 (PR 1 is commit 9e932e7d4); PR_AREA defaults to crv_fiberatten (PR 1 + PR 2, same default output).
# usage: [PR_AREA=...] run_fiberspeed_identity.sh [NEVTS=4000]
N=${1:-4000}
D=/exp/mu2e/data/users/ram2aq/crv_fastsim/fiberspeed_ab; O=$D/identity
UNP=/exp/mu2e/app/users/ram2aq/crv_febphase; PR=${PR_AREA:-/exp/mu2e/app/users/ram2aq/crv_fiberatten}
M=/exp/mu2e/app/users/ram2aq/analysis/mu2e_track_crv_mcresponse/fcl
mkdir -p "$O" && cd "$O" || exit 2
sed 's/dig.nominal.art/dig.unpatched.art/' "$D/fcl/digi_nominal.fcl" > digi_unpatched.fcl
run() {  # run <area> <log> <mu2e args...>
  local area=$1 log=$2; shift 2
  ( source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1; cd "$area" && muse setup >/dev/null 2>&1
    cd "$O" && FHICL_FILE_PATH="$O:$M:$FHICL_FILE_PATH:$UNP:$UNP/build/al9-prof-e29-p108" mu2e "$@" > "$log" 2>&1 )
  echo "$log rc=$? $(grep -m1 'Events total' "$log")"
}
run "$UNP" digi_unpatched.log -c digi_unpatched.fcl -s "$D/dts.art" -n "$N"
run "$PR"  reco_unpatched.log -c reco_crvonly_extracted_v05.fcl -s dig.unpatched.art -o mcs.unpatched.art
run "$UNP" nt_unpatched.log   -c ntuple_crvonly_v05.fcl -s mcs.unpatched.art -T nts.unpatched.root
rm -f dig.unpatched.art
