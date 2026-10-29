#!/bin/bash
# Resolve one documentary worker-log candidate without reading physics payloads.
set -euo pipefail
ROOT=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028/ce_reference_evidence_v1
OUT=${1:?supply a new output directory beneath the reference evidence root}
case "$OUT" in "$ROOT"/*) ;; *) echo 'Unexpected output root' >&2; exit 2 ;; esac
case "$OUT" in *'/../'*|*'/./'*|*'/..'|*'/.' ) exit 2 ;; esac
test "$(hostname -s)" = mu2egpvm05
test "$(cat /exp/mu2e/data/users/ram2aq/agent_locks/mu2egpvm05/task)" = 'veto ce_reference_evidence_v1 20261028'
mkdir "$OUT"
cp "$0" "$OUT/capture.sh"
run_query() {
  local tag=$1 rc bytes
  shift
  printf '%q ' "$@" > "$OUT/$tag.command.txt"
  printf '\n' >> "$OUT/$tag.command.txt"
  set +e
  timeout 60 "$@" 2>&1 | head -c 262145 > "$OUT/$tag.txt"
  rc=${PIPESTATUS[0]}
  set -e
  bytes=$(wc -c < "$OUT/$tag.txt")
  printf '%s\t%s\t%s\n' "$tag" "$rc" "$bytes" >> "$OUT/status.tsv"
  test "$bytes" -le 262144 || { echo 'Query byte cap exceeded' >&2; exit 2; }
  test "$rc" -eq 0 || { echo 'Query failed; stop, no retry' >&2; exit "$rc"; }
}
SAM=(samweb --max-timeout=45 --socket-timeout=20)
run_query worker_names "${SAM[@]}" list-files 'file_name like log.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.0-%.log with limit 2'
test "$(wc -l < "$OUT/worker_names.txt")" -eq 1
NAME=$(cat "$OUT/worker_names.txt")
[[ "$NAME" =~ ^log\.mu2e\.CeMLeadingLogMix1BB\.MDC2025au_best_v1_3\.0-[0-9]+\.log$ ]]
run_query worker_metadata "${SAM[@]}" get-metadata "$NAME"
run_query worker_location "${SAM[@]}" locate-file "$NAME"
run_query known_definition "${SAM[@]}" describe-definition dig.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.art
run_query exact_definition_names "${SAM[@]}" list-definitions --defname=dig.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.art
TOTAL=$(wc -c "$OUT"/*.txt | tail -1 | awk '{print $1}')
test "$TOTAL" -le 1048576
cmp "$0" "$OUT/capture.sh"
printf '%s\n' 'One candidate log catalog record only; no worker/output mapping or independent calibration certified.' > "$OUT/scope.txt"
(cd "$OUT" && sha256sum capture.sh *.command.txt status.tsv scope.txt worker_names.txt worker_metadata.txt worker_location.txt known_definition.txt exact_definition_names.txt > files_sha256.txt)
