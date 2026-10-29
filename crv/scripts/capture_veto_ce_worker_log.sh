#!/bin/bash
# Capture one size-capped, catalog-hash-bound worker log at its exact known path.
set -euo pipefail
ROOT=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028/ce_reference_evidence_v1
OUT=${1:?supply a new output directory beneath the reference evidence root}
case "$OUT" in "$ROOT"/*) ;; *) exit 2 ;; esac
case "$OUT" in *'/../'*|*'/./'*|*'/..'|*'/.' ) exit 2 ;; esac
test "$(hostname -s)" = mu2egpvm05
test "$(cat /exp/mu2e/data/users/ram2aq/agent_locks/mu2egpvm05/task)" = 'veto ce_reference_evidence_v1 20261028'
NAME=log.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.0-1785087048.log
INPUT=/pnfs/mu2e/persistent/datasets/phy-etc/log/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_3/log/e3/7c/$NAME
SHA=dc2b977ec44bf477c0aa68f7f124ca863b9fb29eac41f53b14ee843c37b35143
mkdir "$OUT"
cp "$0" "$OUT/capture.sh"
printf '%s\n' "$INPUT" > "$OUT/input_path.txt"
printf '%s\n' 'Catalog bytes=449780; read cap=449781; timeout=60s; no retries or namespace metadata calls.' > "$OUT/read_contract.txt"
set +e
timeout 60 head -c 449781 "$INPUT" > "$OUT/worker.log" 2> "$OUT/read.stderr"
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/read.status"
test "$RC" -eq 0
test "$(wc -c < "$OUT/worker.log")" -eq 449780
printf '%s  worker.log\n' "$SHA" > "$OUT/expected_sha256.txt"
(cd "$OUT" && sha256sum -c expected_sha256.txt)
cmp "$0" "$OUT/capture.sh"
printf '%s\n' 'Exact worker-log bytes verified; output mapping and historical deployment require reading this record, not filename inference.' > "$OUT/scope.txt"
(cd "$OUT" && sha256sum capture.sh input_path.txt read_contract.txt read.stderr read.status expected_sha256.txt worker.log scope.txt > files_sha256.txt)
