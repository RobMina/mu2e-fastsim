#!/bin/bash
# Capture bounded read-only catalog leads, never physics payloads or credentials.
set -euo pipefail
ROOT=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028/ce_reference_evidence_v1
OUT=${1:?supply a new output directory beneath the reference evidence root}
case "$OUT" in "$ROOT"/*) ;; *) echo 'Unexpected output root' >&2; exit 2 ;; esac
test "$(hostname -s)" = mu2egpvm05
test "$(cat /exp/mu2e/data/users/ram2aq/agent_locks/mu2egpvm05/task)" = 'veto ce_reference_evidence_v1 20261028'
mkdir "$OUT"
cp "$0" "$OUT/discover.sh"
run_query() {
  local tag=$1 rc bytes
  shift
  printf '%q ' "$@" > "$OUT/$tag.command.txt"
  printf '\n' >> "$OUT/$tag.command.txt"
  set +e
  "$@" 2>&1 | head -c 262145 > "$OUT/$tag.txt"
  rc=${PIPESTATUS[0]}
  set -e
  bytes=$(wc -c < "$OUT/$tag.txt")
  printf '%s\t%s\t%s\n' "$tag" "$rc" "$bytes" >> "$OUT/status.tsv"
  test "$bytes" -le 262144 || { echo 'Query byte cap exceeded' >&2; exit 2; }
  test "$rc" -eq 0 || { echo 'Query failed; stop, no retry' >&2; exit "$rc"; }
}
SAM=(samweb --max-timeout=45 --socket-timeout=20)
run_query flatpbi_definitions "${SAM[@]}" list-definitions --defname='*FlatPBI*'
run_query luminosity_definitions "${SAM[@]}" list-definitions --defname='*Luminosity*'
run_query intensity_definitions "${SAM[@]}" list-definitions --defname='*Intensity*'
run_query ce_mix1bb_definitions "${SAM[@]}" list-definitions --defname='*CeMLeadingLogMix1BB*'
run_query dig_worker0_count "${SAM[@]}" count-files 'file_name like log.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.0-%.log'
run_query reco_worker0_count "${SAM[@]}" count-files 'file_name like log.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.0-%.log'
run_query ntuple_worker0_count "${SAM[@]}" count-files 'file_name like log.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.0-%.log'
TOTAL=$(wc -c "$OUT"/*.txt | tail -1 | awk '{print $1}')
test "$TOTAL" -le 1048576
cmp "$0" "$OUT/discover.sh"
printf '%s\n' 'Catalog-name discovery only; no independent reference, runtime mapping or sample completeness certified.' > "$OUT/scope.txt"
(cd "$OUT" && sha256sum discover.sh *.command.txt status.tsv scope.txt flatpbi_definitions.txt luminosity_definitions.txt intensity_definitions.txt ce_mix1bb_definitions.txt dig_worker0_count.txt reco_worker0_count.txt ntuple_worker0_count.txt > files_sha256.txt)
