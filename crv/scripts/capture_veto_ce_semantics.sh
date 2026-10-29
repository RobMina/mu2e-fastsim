#!/bin/bash
# Capture bounded documentary sources; never execute a production configuration.
set -euo pipefail
OUT=${1:?supply a new absolute output directory}
ROOT=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028
case "$OUT" in "$ROOT"/*) ;; *) echo 'Output outside dedicated evidence area' >&2; exit 2 ;; esac
mkdir "$OUT"
mkdir "$OUT/sources" "$OUT/catalog" "$OUT/jobdef"
cp "$0" "$OUT/capture.sh"
DIG=/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025au
OFF=$DIG/backing/Offline
OLD=$ROOT/historical_config_v1
F=/pnfs/mu2e/persistent/datasets/phy-etc/cnf/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_3/tar/a5/43/cnf.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.0.tar
printf '%s  %s\n' b747fde8badafff79918a20e6a6bf2ea534351c549ec6a73e6d7b1dffd3dc1bb "$F" > "$OUT/jobdef/expected_sha256.txt"
sha256sum -c "$OUT/jobdef/expected_sha256.txt"
tar -tvf "$F" > "$OUT/jobdef/members.txt"
tar -xOf "$F" jobpars.json | head -c 1048577 > "$OUT/jobdef/jobpars.json"
tar -xOf "$F" mu2e.fcl | head -c 8193 > "$OUT/jobdef/mu2e.fcl"
test "$(wc -c < "$OUT/jobdef/jobpars.json")" -eq 254851
test "$(wc -c < "$OUT/jobdef/mu2e.fcl")" -eq 519
jq '{setup,jobname,event_id:.tbs.event_id,subrunkey:.tbs.subrunkey,seed:.tbs.seed,njobs:.tbs.njobs,input_keys:(.tbs.inputs|keys),auxiliary_keys:(.tbs.auxin|keys)}' "$OUT/jobdef/jobpars.json" > "$OUT/jobdef/summary.json"
TOTAL=0
snapshot() {
  local input=$1 output=$2 size before
  size=$(wc -c < "$input")
  test "$size" -le 1048576
  TOTAL=$((TOTAL+size))
  test "$TOTAL" -le 16777216
  before=$(sha256sum "$input" | cut -d ' ' -f 1)
  cp "$input" "$OUT/$output"
  test "$(sha256sum "$OUT/$output" | cut -d ' ' -f 1)" = "$before"
  printf '%s  %s\n' "$before" "$input" >> "$OUT/originals_sha256.txt"
  printf '%s\t%s\n' "$output" "$input" >> "$OUT/source_paths.tsv"
}
for n in ce_mcs_parent_metadata_v1 ce_dig_parent_metadata_v1 ce_dig_jobdef_discovery_v1 ce_dig_jobdef_metadata_v1 ce_dig_jobdef_location_v1; do
  snapshot "$ROOT/$n.log" "catalog/$n.log"
done
for n in ce_prolog.fcl ce_from_mcs-mockdata.fcl ce_EventNtupleMaker_module.cc ce_InfoStructHelper.cc ce_EventInfo.hh ce_LumiStreamInfo.hh; do
  snapshot "$OLD/sources/$n" "sources/$n"
done
snapshot "$DIG/setup.sh" sources/dig_setup.sh
for n in mixing/Mix.fcl mixing/OneBB.fcl mixing/prolog.fcl mixing/epilog.fcl digitize/prolog.fcl digitize/epilog.fcl common/epilog.fcl; do
  snapshot "$DIG/Production/JobConfig/$n" "sources/production_${n//\//_}"
done
for n in core/trigProducers.fcl core/trigRecoSequences.fcl core/trigDigiInputsEpilog.fcl core/producers/trigTrkProducers.fcl core/producers/trigCalProducers.fcl; do
  snapshot "$DIG/mu2e-trig-config/$n" "sources/trigger_${n//\//_}"
done
for n in CommonMC/fcl/prolog.fcl CommonMC/src/EventWindowMarkerProducer_module.cc RecoDataProducts/inc/ProtonBunchTime.hh MCDataProducts/inc/ProtonBunchTimeMC.hh TrkHitReco/fcl/prolog.fcl TrkHitReco/src/StrawHitReco_module.cc TrkHitReco/src/StrawHitRecoUtils.cc CalPatRec/fcl/prolog.fcl CalPatRec/src/DeltaFinder_module.cc CalPatRec/src/DeltaFinderAlg.cc CalPatRec/inc/DeltaFinderAlg.hh CalPatRec/src/ProtonCandidate.cc CalPatRec/inc/ProtonCandidate.hh CaloReco/fcl/common.fcl CaloReco/src/CaloHitMakerFast_module.cc RecoDataProducts/inc/IntensityInfoTimeCluster.hh RecoDataProducts/inc/IntensityInfoTrackerHits.hh RecoDataProducts/inc/IntensityInfoCalo.hh; do
  snapshot "$OFF/$n" "sources/offline_${n//\//_}"
done
SEL=/exp/mu2e/data/users/ram2aq/crv_fastsim/deadtime_run1a
ANA=/exp/mu2e/data/users/ram2aq/crv_fastsim/mu2e_cosmic_ana
snapshot "$SEL/select_ce_mix1bb.sh" sources/saved_select_ce_mix1bb.sh
snapshot "$SEL/select_ce_mix1bb.log" sources/saved_select_ce_mix1bb.log
snapshot "$ANA/src/core/analyse.py" sources/available_analyse.py
snapshot "$ANA/config/common/cuts.yaml" sources/available_cuts.yaml
snapshot /exp/mu2e/app/users/ram2aq/analysis/crv_wt/r2_veto/crvfastsim/veto/cosmic_ana.py sources/available_cosmic_ana.py
printf '%s\t%s\n' "$DIG/backing" "$(readlink -f "$DIG/backing")" > "$OUT/backing.tsv"
sha256sum -c "$OUT/originals_sha256.txt"
sha256sum -c "$OUT/jobdef/expected_sha256.txt"
cmp "$0" "$OUT/capture.sh"
printf '%s\n' 'Documentary recipe/source evidence, not a resolved worker configuration or complete historical execution certificate.' > "$OUT/scope.txt"
(cd "$OUT" && find . -type f ! -name files_sha256.txt -print0 | sort -z | xargs -0 sha256sum > files_sha256.txt)
printf 'captured source/catalog bytes=%s\n' "$TOTAL"
