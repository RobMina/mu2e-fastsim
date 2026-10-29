#!/bin/bash
# Preserve authenticated KB captures, including the explicitly failed search.
set -euo pipefail
ROOT=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028/ce_kb_evidence_v1
WORK=/exp/mu2e/app/users/ram2aq/analysis/crv_wt/r2_veto
test "$(hostname -s)" = mu2eaigpvm01
test "$(cat /exp/mu2e/data/users/ram2aq/agent_locks/mu2eaigpvm01/task)" = 'veto ce_kb_evidence_v1 20261028'
test ! -e "$ROOT/finalized_files_sha256.txt"
mkdir "$ROOT/completed_docs"
cp "$0" "$ROOT/completed_docs/finalize.sh"
SEARCHES=(search_luminosity search_ihep_exact search_bunch_time search_pbi_model search_timing_doc search_werkema)
GETS=(get_lumi_status get_lumi_analysis get_rmc_normalization get_tracker_normalization get_pbi_review get_intensity_flow get_normalization_strategy get_mdc2020_note)
for tag in search_ihep "${SEARCHES[@]}" "${GETS[@]}"; do
  (cd "$ROOT/$tag" && sha256sum -c files_sha256.txt) >> "$ROOT/completed_docs/child_checks.txt"
  if test "$tag" = search_ihep; then
    test "$(cat "$ROOT/$tag/status.txt")" = '124 0'
    test ! -s "$ROOT/$tag/result.txt"
  else
    test "$(cat "$ROOT/$tag/status.txt")" = '0 0'
  fi
done
for tag in "${SEARCHES[@]}"; do
  jq -e '.count == (.results|length) and .count > 0 and .count <= 4' "$ROOT/$tag/result.txt" > /dev/null
done
for tag in "${GETS[@]}"; do
  grep -q '^\[\[DOCUMENT_END\]\]$' "$ROOT/$tag/result.txt"
  test "$(head -n 1 "$ROOT/$tag/result.txt")" = '[[DOCUMENT_METADATA]]'
done
test "$(cat "$ROOT/tools_list.status")" = 0
snapshot() {
  local input=$1 expected=$2 output=$3
  test "$(sha256sum "$input" | cut -d ' ' -f 1)" = "$expected"
  cp "$input" "$ROOT/completed_docs/$output"
  test "$(sha256sum "$ROOT/completed_docs/$output" | cut -d ' ' -f 1)" = "$expected"
  printf '%s  %s\n' "$expected" "$input" >> "$ROOT/completed_docs/originals_sha256.txt"
}
snapshot "$WORK/docs/veto-ce-kb-evidence.md" 7762d6a38d1505bade69229b3625b6baa2a32478b610488d349690d80621e5cf veto-ce-kb-evidence.md
snapshot "$WORK/docs/veto-continuation.md" cfa62023b321be1ceba5c3d37464321570ebfbfe7f9c20b747b536d0f58b3591 veto-continuation.md
snapshot "$WORK/scripts/capture_veto_ce_kb.sh" b0c370fa3f9f377d1ce86c56db75781235108ed141c4e4fb12d611d53063e3e0 capture_veto_ce_kb.sh
bash -n "$WORK/scripts/capture_veto_ce_kb.sh"
printf '%s\n' 'Authentication successful. Six searches/eight gets successful; one search timeout124, no retry. Failed wrapper completion line is not success. No independent calibration admitted.' > "$ROOT/completed_docs/status.txt"
sha256sum -c "$ROOT/completed_docs/originals_sha256.txt" > "$ROOT/completed_docs/originals_check.txt"
cmp "$0" "$ROOT/completed_docs/finalize.sh"
(cd "$ROOT" && find . -type f ! -path ./finalized_files_sha256.txt -print0 | sort -z | xargs -0 sha256sum > finalized_files_sha256.txt)
(cd "$ROOT" && sha256sum -c finalized_files_sha256.txt)
sha256sum "$ROOT/finalized_files_sha256.txt"
find "$ROOT" -type f -printf '%s\n' | awk '{s+=$1;n++} END {printf "files=%d bytes=%d including final manifest\n",n,s}'
