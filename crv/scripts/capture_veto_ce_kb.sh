#!/bin/bash
# Capture bounded read-only KB evidence through the documented on-host client.
set -euo pipefail
ROOT=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028/ce_kb_evidence_v1
TAG=${1:?supply a new capture tag}
TOOL=${2:?kb_search or kb_get}
ARGS=${3:?supply JSON arguments}
[[ "$TAG" =~ ^[a-z][a-z0-9_]{0,63}$ ]]
case "$TOOL" in kb_search|kb_get) ;; *) exit 2 ;; esac
test "$(hostname -s)" = mu2eaigpvm01
test "$(cat /exp/mu2e/data/users/ram2aq/agent_locks/mu2eaigpvm01/task)" = 'veto ce_kb_evidence_v1 20261028'
CLIENT=/exp/mu2e/app/users/ram2aq/goose/secrets/kbq.py
test "$(sha256sum "$CLIENT" | cut -d ' ' -f 1)" = c51f2b03085b0744f31453659905f672348bca9e69be0a8f843b112df1bb7687
OUT=$ROOT/$TAG
mkdir "$OUT"
cp "$0" "$OUT/capture.sh"
printf '%s\n' "$TOOL" > "$OUT/tool.txt"
printf '%s\n' "$ARGS" > "$OUT/arguments.json"
printf '%s\n' 'Documented kbq.py; credentials stay on host; 120s timeout, 256MiB address space, 1MiB stdout cap; no retries.' > "$OUT/contract.txt"
ulimit -v 262144
set +e
/usr/bin/time -v -o "$OUT/time.txt" timeout 120 python3 "$CLIENT" "$TOOL" "$ARGS" 2> "$OUT/stderr.txt" | head -c 1048577 > "$OUT/result.txt"
STATUS=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${STATUS[*]}" > "$OUT/status.txt"
if test "${STATUS[0]}" -ne 0 || test "${STATUS[1]}" -ne 0; then
  echo 'Client/pipeline failed; preserve capture and stop without retry' >&2
  exit 2
fi
test "$(wc -c < "$OUT/result.txt")" -le 1048576
cmp "$0" "$OUT/capture.sh"
(cd "$OUT" && sha256sum capture.sh tool.txt arguments.json contract.txt time.txt stderr.txt result.txt status.txt > files_sha256.txt)
printf '%s: %s bytes\n' "$TAG" "$(wc -c < "$OUT/result.txt")"
