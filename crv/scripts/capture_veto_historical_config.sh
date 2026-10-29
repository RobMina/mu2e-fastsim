#!/bin/bash
# Archive documentary evidence only; never execute a production configuration.
set -euo pipefail

OUT=${1:?supply a new absolute output directory under the data area}
case "$OUT" in
  /exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028/*) ;;
  *) echo 'Output must be under the dedicated historical evidence data area' >&2; exit 2 ;;
esac
mkdir "$OUT"
mkdir "$OUT/sources" "$OUT/jobdefs" "$OUT/catalog"
cp "$0" "$OUT/capture.sh"

CVM=/cvmfs/mu2e.opensciencegrid.org
DATA=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028
FLASH=/pnfs/mu2e/persistent/datasets/phy-etc/cnf/mu2e/NeutralsFlash/MDC2025ac/tar/6f/cc/cnf.mu2e.NeutralsFlash.MDC2025ac.0.tar
CAT=/pnfs/mu2e/persistent/datasets/phy-etc/cnf/mu2e/NeutralsFlashCat/MDC2025ad/tar/ad/f4/cnf.mu2e.NeutralsFlashCat.MDC2025ad.0.tar
CE=/pnfs/mu2e/persistent/datasets/phy-etc/cnf/mu2e/evnt/MDC2025au_best_v1_1-001/tar/a5/58/cnf.mu2e.evnt.MDC2025au_best_v1_1-001.0.tar

cat > "$OUT/jobdefs/expected_sha256.txt" <<EOF
8785cba6fcbd6590ef8ec58035c9cd15966cb7bfb0849cf344bb9086dc1b0e9a  $FLASH
c17855fe607c8233b160e1158767bee5b4f9c80241da1196201297c80b3651fb  $CAT
863ae9650e84613a78d9e2b82b8fb15864ae74df89f628dd1c51e749d4728f18  $CE
EOF
sha256sum -c "$OUT/jobdefs/expected_sha256.txt"

capture_member() {
  local file=$1 member=$2 output=$3 cap=$4
  tar -xOf "$file" "$member" | head -c "$((cap+1))" > "$output"
  test "$(wc -c < "$output")" -le "$cap"
}

for name in FLASH CAT CE; do
  file=${!name}
  tar -tvf "$file" > "$OUT/jobdefs/$name.members.txt"
  capture_member "$file" mu2e.fcl "$OUT/jobdefs/$name.mu2e.fcl" 8192
  if test "$name" != CAT; then
    capture_member "$file" jobpars.json "$OUT/jobdefs/$name.jobpars.json" 65536
    jq -e '(.setup | type == "string") and (.tbs | type == "object")' "$OUT/jobdefs/$name.jobpars.json" >/dev/null
  fi
done
printf '%s\n' 'CAT jobpars.json was not decoded or extracted: declared member size 360000549 bytes.' > "$OUT/jobdefs/CAT.scope.txt"

cat > "$OUT/catalog/expected_sha256.txt" <<EOF
33dc7a337f6140c7a26ad673f6e9bf18f81a6f880ac8bec5cc78986b82fb6201  $DATA/historical_parent_metadata_v1.log
ccea5300f3b602f1cfa064088a42bffdb9fd0055fce92ab16d36adb7d7d2fa60  $DATA/historical_flash_sibling_metadata_v1.log
c14cad84a040cc48b09ece6818a376509f99f981698fa128c1c40ae2b9237548  $DATA/historical_upstream_metadata_v1.log
c5157bcf05f879813842cf9393536f4ec60690143e723f7c66251ab1484893a5  $DATA/historical_ce_metadata_v1.log
EOF
sha256sum -c "$OUT/catalog/expected_sha256.txt"
for name in historical_parent_metadata_v1 historical_flash_sibling_metadata_v1 historical_upstream_metadata_v1 historical_ce_metadata_v1 historical_jobdef_metadata_v1 historical_jobdef_discovery_v2; do
  test "$(wc -c < "$DATA/$name.log")" -le 1048576
  cp "$DATA/$name.log" "$OUT/catalog/$name.log"
done

snapshot() {
  local source=$1 target=$2 before
  test "$(wc -c < "$source")" -le 1048576
  before=$(sha256sum "$source" | cut -d ' ' -f 1)
  cp "$source" "$OUT/sources/$target"
  test "$(sha256sum "$OUT/sources/$target" | cut -d ' ' -f 1)" = "$before"
  printf '%s  %s\n' "$before" "$source" >> "$OUT/source_original_sha256.txt"
  printf '%s\t%s\n' "$target" "$source" >> "$OUT/source_paths.tsv"
}

FLASH_RELEASE=$CVM/Musings/SimJob/MDC2025ac
CAT_RELEASE=$CVM/Musings/SimJob/MDC2025ad
NTUPLE=$CVM/Musings/AnalysisMDC2025/v02_01_00/EventNtuple
snapshot "$FLASH_RELEASE/setup.sh" flash_setup.sh
snapshot "$FLASH_RELEASE/Production/JobConfig/pileup/NeutralsResampler.fcl" flash_NeutralsResampler.fcl
snapshot "$FLASH_RELEASE/Production/JobConfig/pileup/prolog.fcl" flash_pileup_prolog.fcl
snapshot "$FLASH_RELEASE/Production/JobConfig/common/prolog.fcl" flash_common_prolog.fcl
snapshot "$FLASH_RELEASE/backing/Offline/EventMixing/src/ResamplingMixer_module.cc" flash_ResamplingMixer_module.cc
snapshot "$FLASH_RELEASE/backing/Offline/EventGenerator/src/GenEventCounter_module.cc" flash_GenEventCounter_module.cc
snapshot "$FLASH_RELEASE/backing/Offline/CommonMC/src/GenEventCountReader_module.cc" flash_GenEventCountReader_module.cc
snapshot "$CAT_RELEASE/setup.sh" cat_setup.sh
snapshot "$CAT_RELEASE/Production/JobConfig/common/artcat.fcl" cat_artcat.fcl
snapshot "$CAT_RELEASE/backing/Offline/CommonMC/src/GenEventCountReader_module.cc" cat_GenEventCountReader_module.cc
snapshot "$CVM/Musings/AnalysisMDC2025/v02_01_00/setup.sh" ce_setup.sh
snapshot "$NTUPLE/fcl/from_mcs-mockdata.fcl" ce_from_mcs-mockdata.fcl
snapshot "$NTUPLE/fcl/prolog.fcl" ce_prolog.fcl
snapshot "$NTUPLE/fcl/prolog_mc_reweighters.fcl" ce_prolog_mc_reweighters.fcl
snapshot "$NTUPLE/src/EventNtupleMaker_module.cc" ce_EventNtupleMaker_module.cc
snapshot "$NTUPLE/src/InfoStructHelper.cc" ce_InfoStructHelper.cc
snapshot "$NTUPLE/inc/EventInfo.hh" ce_EventInfo.hh
snapshot "$NTUPLE/inc/LumiStreamInfo.hh" ce_LumiStreamInfo.hh
snapshot "$NTUPLE/README.md" ce_README.md
snapshot "$CVM/artexternals/mu2efilename/v3_13/bin/printJson.sh" candidate_printJson.sh
snapshot "$CVM/artexternals/mu2egrid/v8_03_02/bin/impl/mu2eprodsys.sh" candidate_mu2eprodsys.sh
snapshot "$CVM/bin/prodtools/v1.3.5/utils/jobfcl.py" candidate_jobfcl_v1.3.5.py
snapshot "$CVM/bin/prodtools/v3.3.4/utils/job_common.py" candidate_job_common_v3.3.4.py
snapshot "$CVM/bin/prodtools/v3.3.4/data/mdc2025/resampler_beam.json" candidate_resampler_beam_v3.3.4.json
snapshot "$CVM/bin/prodtools/v3.3.4/data/mdc2025/evntuple.json" candidate_evntuple_v3.3.4.json
for release in "$FLASH_RELEASE" "$CAT_RELEASE" "$CVM/Musings/AnalysisMDC2025/v02_01_00"; do
  printf '%s\t%s\n' "$release/backing" "$(readlink -f "$release/backing")" >> "$OUT/backing_paths.tsv"
done
sha256sum -c "$OUT/source_original_sha256.txt"
sha256sum -c "$OUT/jobdefs/expected_sha256.txt"
sha256sum -c "$OUT/catalog/expected_sha256.txt"
cmp "$0" "$OUT/capture.sh"
(cd "$OUT" && find . -type f ! -name files_sha256.txt ! -name complete.txt -print0 | sort -z | xargs -0 sha256sum > files_sha256.txt)
printf '%s\n' 'Documentary capture complete; not a production-execution certificate.' > "$OUT/complete.txt"
