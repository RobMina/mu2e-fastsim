#!/bin/bash
# Profile the stock extracted cosmic chain (sim then digi) on one GPVM with TimeTracker/MemoryTracker.
# usage: run_profile.sh <stage: sim|digi|dumps> <nevents> [input.art for digi]   (needs site.env: CRVFS_CODE, CRVFS_DATA)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup SimJob MDC2025ay >/dev/null 2>&1 || true
set -u

HERE=${CRVFS_CODE:?source site.env}/profile
OUT=${CRVFS_DATA:?source site.env}/profile
mkdir -p "$OUT"
cd "$OUT"
export FHICL_FILE_PATH="$HERE/fcl:$FHICL_FILE_PATH"

stage=$1; nev=$2
echo "host=$(hostname -s) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) start=$(date -Is)"
echo "offline=$(which mu2e)"
case "$stage" in
  sim)
    /usr/bin/time -v mu2e -c prof_sim_extracted.fcl -n "$nev" > sim.log 2>&1
    echo "exit=$?" | tee -a sim.log ;;
  digi)
    in=${3:-$(ls dts.*.CosmicCRYExtractedProf.v0.001400_00000001.art | head -1)}
    /usr/bin/time -v mu2e -c prof_digi_extracted.fcl -s "$in" -n "$nev" > digi.log 2>&1
    echo "exit=$?" | tee -a digi.log ;;
  dumps)
    for f in dts.*.art dig.*.art; do
      product_sizes_dumper -f 1000 "$f" > "sizes_${f%%.*}.txt" 2>&1
      file_info_dumper --event-list "$f" 2>/dev/null | grep -c '^ *[0-9]' > "nevents_${f%%.*}.txt"
    done ;;
esac
echo "end=$(date -Is)"
