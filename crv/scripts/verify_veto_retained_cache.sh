#!/usr/bin/env bash
# Verify saved cache audit and selected tests without replay loading or physics generation.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 EXISTING_CACHE_AUDIT EXPECTED_PLAN_SHA256 NEW_OUTPUT" >&2
  exit 2
fi
existing=$1
plan_sha=$2
out=$3
mkdir "$out"
python3 -m crvfastsim.veto.retained_cache_report "$existing" --expected-plan-sha256 "$plan_sha" \
  --out "$out/verified.json" --compact > "$out/compact.json"
cat "$out/compact.json"
python3 -m mu2efastsim.testselect --why > "$out/selection.txt"
python3 -m mu2efastsim.testselect --run --isolate -- -q -s > "$out/selected_tests.txt" 2>&1
cat "$out/selected_tests.txt"
for name in veto_replay_timing veto_replay; do
  python3 -m pytest -p no:cacheprovider -q "tests/test_${name}.py" > "$out/test_${name}.txt" 2>&1
  cat "$out/test_${name}.txt"
done
python3 --version > "$out/python_version.txt"
mkdir "$out/code"
for file in crvfastsim/veto/retained_cache_audit.py crvfastsim/veto/retained_cache_report.py \
            tests/test_veto_retained_cache_audit.py tests/test_veto_retained_cache_report.py \
            docs/veto-retained-cache-audit.md scripts/verify_veto_retained_cache.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
printf '%s\n' 'Saved cache evidence only, no retained-cache reload, optics/response generation or campaign admission.' > "$out/scope.txt"
