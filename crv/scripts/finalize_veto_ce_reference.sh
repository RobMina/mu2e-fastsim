#!/bin/bash
# Finalize documentary evidence in place without regenerating successful captures.
set -euo pipefail
ROOT=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028/ce_reference_evidence_v1
WORK=/exp/mu2e/app/users/ram2aq/analysis/crv_wt/r2_veto
test "$(hostname -s)" = mu2egpvm05
test "$(cat /exp/mu2e/data/users/ram2aq/agent_locks/mu2egpvm05/task)" = 'veto ce_reference_evidence_v1 20261028'
test ! -e "$ROOT/finalized_files_sha256.txt"
mkdir "$ROOT/completed_docs"
cp "$0" "$ROOT/completed_docs/finalize.sh"
for d in catalog_discovery_v1 worker_catalog_v1 worker_log_v1 retained_worker12_v1; do
  (cd "$ROOT/$d" && sha256sum -c files_sha256.txt) > "$ROOT/completed_docs/${d}_check.txt"
done
snapshot() {
  local input=$1 expected=$2 output=$3
  test "$(sha256sum "$input" | cut -d ' ' -f 1)" = "$expected"
  cp "$input" "$ROOT/completed_docs/$output"
  test "$(sha256sum "$ROOT/completed_docs/$output" | cut -d ' ' -f 1)" = "$expected"
  printf '%s  %s\n' "$expected" "$input" >> "$ROOT/completed_docs/originals_sha256.txt"
}
snapshot "$WORK/docs/veto-ce-reference-evidence.md" b7dbd3eed1be9c8dbdd1e0ba4dc2b048ef9753bfaaac5c4ac2704fe0e8fc6e65 veto-ce-reference-evidence.md
snapshot "$WORK/docs/veto-continuation.md" 801e0f21957ed557d53105a1b189db2746745edf7cfb0fc612cc60218435629c veto-continuation.md
for name in discover_veto_ce_reference capture_veto_ce_worker_catalog capture_veto_ce_worker_log capture_veto_ce_retained_worker; do
  bash -n "$WORK/scripts/$name.sh"
done
sed -n '27,113p;125p;136,137p;237,242p;600p;2643p;2848p;2927,2937p' "$ROOT/retained_worker12_v1/worker.log" > "$ROOT/completed_docs/worker12_documentary_excerpt.txt"
printf '%s\n' 'Original worker-log line numbers are specified in veto-ce-reference-evidence.md; excerpt omits later environment dump. No physics hashes or calibration validation claimed.' > "$ROOT/completed_docs/excerpt_scope.txt"
sha256sum -c "$ROOT/completed_docs/originals_sha256.txt" > "$ROOT/completed_docs/originals_check.txt"
cmp "$0" "$ROOT/completed_docs/finalize.sh"
(cd "$ROOT" && find . -type f ! -path ./finalized_files_sha256.txt -print0 | sort -z | xargs -0 sha256sum > finalized_files_sha256.txt)
(cd "$ROOT" && sha256sum -c finalized_files_sha256.txt)
sha256sum "$ROOT/finalized_files_sha256.txt"
find "$ROOT" -type f -printf '%s\n' | awk '{s+=$1;n++} END {printf "files=%d bytes=%d including final manifest\n",n,s}'
