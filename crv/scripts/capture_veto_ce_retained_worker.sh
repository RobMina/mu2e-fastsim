#!/bin/bash
# Follow the saved four-input group12 hypothesis with bounded documentary reads.
set -euo pipefail
ROOT=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028
OUT=${1:?supply a new output directory beneath the reference evidence root}
case "$OUT" in "$ROOT/ce_reference_evidence_v1/"*) ;; *) exit 2 ;; esac
case "$OUT" in *'/../'*|*'/./'*|*'/..'|*'/.' ) exit 2 ;; esac
test "$(hostname -s)" = mu2egpvm05
test "$(cat /exp/mu2e/data/users/ram2aq/agent_locks/mu2egpvm05/task)" = 'veto ce_reference_evidence_v1 20261028'
mkdir "$OUT"
cp "$0" "$OUT/capture.sh"
OLD=$ROOT/ce_proxy_semantics_v1
(cd "$OLD" && sha256sum -c files_sha256.txt) > "$OUT/prior_capture_check.txt"
jq '{setup,group_size:.tbs.inputs["source.fileNames"][0],group12:.tbs.inputs["source.fileNames"][1][48:52]}' "$OLD/jobdef/jobpars.json" > "$OUT/group12.json"
test "$(jq '.group_size' "$OUT/group12.json")" = 4
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
  test "$bytes" -le 262144
  test "$rc" -eq 0 || { echo 'Query failed; no retry' >&2; exit "$rc"; }
}
SAM=(samweb --max-timeout=45 --socket-timeout=20)
run_query worker_names "${SAM[@]}" list-files 'file_name like log.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.12-%.log with limit 2'
test "$(wc -l < "$OUT/worker_names.txt")" -eq 1
NAME=$(cat "$OUT/worker_names.txt")
[[ "$NAME" =~ ^log\.mu2e\.CeMLeadingLogMix1BB\.MDC2025au_best_v1_3\.12-[0-9]+\.log$ ]]
run_query worker_metadata "${SAM[@]}" get-metadata "$NAME"
run_query worker_location "${SAM[@]}" locate-file "$NAME"
SIZE=$(sed -n 's/^ *File Size: *//p' "$OUT/worker_metadata.txt")
SHA=$(sed -n 's/^ *dh.sha256: *//p' "$OUT/worker_metadata.txt")
LOCATION=$(cat "$OUT/worker_location.txt")
[[ "$SIZE" =~ ^[0-9]+$ ]] && test "$SIZE" -gt 0 && test "$SIZE" -le 1048576
[[ "$SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$LOCATION" =~ ^dcache:/pnfs/mu2e/persistent/datasets/phy-etc/log/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_3/log/[0-9a-f]{2}/[0-9a-f]{2}$ ]]
INPUT=${LOCATION#dcache:}/$NAME
printf '%s\n' "$INPUT" > "$OUT/input_path.txt"
printf 'Catalog bytes=%s; read cap=%s; timeout=60s; no retries or namespace metadata calls.\n' "$SIZE" "$((SIZE+1))" > "$OUT/read_contract.txt"
set +e
timeout 60 head -c "$((SIZE+1))" "$INPUT" > "$OUT/worker.log" 2> "$OUT/read.stderr"
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/read.status"
test "$RC" -eq 0
test "$(wc -c < "$OUT/worker.log")" -eq "$SIZE"
printf '%s  worker.log\n' "$SHA" > "$OUT/expected_sha256.txt"
(cd "$OUT" && sha256sum -c expected_sha256.txt)
cmp "$0" "$OUT/capture.sh"
printf '%s\n' 'Worker12 hypothesis only until log inspection; no physics payload read or independent-reference certification.' > "$OUT/scope.txt"
(cd "$OUT" && find . -maxdepth 1 -type f ! -name files_sha256.txt -print0 | sort -z | xargs -0 sha256sum > files_sha256.txt)
