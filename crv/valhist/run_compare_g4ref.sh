#!/bin/bash
# Merge the g4ref_v1 shard histograms of each arm (shards FIRST..LAST) and compare every arm with its reference, with
# the reseeded floor (A0' vs A0) as the per-histogram scale. usage: run_compare_g4ref.sh FIRST LAST ARM...  (on a GPVM)
#   dts arms (a0p rt a2 a3 a4) against a0; gen-stream arms (a1_*, e2e*) against a0g
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana >/dev/null 2>&1 || true
set -u
A=${1:?first}; B=${2:?last}; shift 2
D=${CRVFS_DATA:?source site.env}/valhist/g4ref_v1; cd "$D"; mkdir -p merged cmp
T=s$(printf %02d "$A")-$(printf %02d "$B")
files() { for k in $(seq "$A" "$B"); do printf "%s/valhist.001400_%08d.root " "$1" "$k"; done; }
AL="--alias CrvSteps_fastCrvSteps_noName=CrvSteps_steps_noName --alias SimParticlemv_fastCrvSteps_noName=SimParticlemv_steps_noName --alias CrvSteps_compressDetStepMCs_noName=CrvSteps_steps_noName"
M() { PYTHONPATH=$CRVFS_CODE:$M2FS_CODE/core python3 -m valhist.compare "$@"; }
for arm in a0 a0p a0g "$@"; do [ -s "merged/$arm.$T.root" ] || M merge "merged/$arm.$T.root" $(files "$arm") $AL | tail -1; done
for arm in "$@"; do
  case $arm in a1*|e2e*|a0g) REF=a0g ;; *) REF=a0 ;; esac
  M compare "merged/$REF.$T.root" "merged/$arm.$T.root" "cmp/${arm}_vs_$REF.$T" --floor "merged/a0.$T.root" \
    "merged/a0p.$T.root" --label-ref "$REF" --label-cand "$arm" --pdf | tail -1
done
