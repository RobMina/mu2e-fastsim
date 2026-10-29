#!/usr/bin/env bash
# Read retained JSON only; no source re-extraction, mixture, response, or stock job.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 REPLAY_BASE PROVENANCE_JSON NEW_OUTPUT_DIR" >&2
  exit 2
fi
base=$1
prior=$2
out=$3
mkdir "$out"
python3 -m crvfastsim.veto.library_weights \
  --manifests "$base/block0/mixed.json" "$base/block1/mixed.json" "$base/block2_fullinventory/mixed.json" \
  --provenance "$prior" --out "$out/weights.json"
python3 -m pytest -q tests/test_veto*.py tests/test_coinc_attachment_cpu.py > "$out/tests.txt" 2>&1
cat "$out/tests.txt"
sha256sum crvfastsim/veto/library_weights.py tests/test_veto_library_weights.py \
  docs/veto-library-weights.md scripts/audit_veto_library_weights.sh > "$out/code_sha256.txt"
