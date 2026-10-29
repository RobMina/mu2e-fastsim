#!/usr/bin/env bash
# Verify saved bounded response and run isolated selected tests without regenerating physics.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 EXISTING_RESPONSE NEW_OUTPUT" >&2
  exit 2
fi
input=$1
out=$2
mkdir "$out"
python3 -m crvfastsim.veto.saved_qualified_report "$input" --out "$out/verified.json" --compact > "$out/compact.json"
python3 -m mu2efastsim.testselect --why > "$out/selection.txt"
python3 -m mu2efastsim.testselect --run --isolate -- -q -s > "$out/selected_tests.txt" 2>&1
cat "$out/selected_tests.txt"
for name in veto_replay_timing veto_replay; do
  python3 -m pytest -p no:cacheprovider -q "tests/test_${name}.py" > "$out/test_${name}.txt" 2>&1
  cat "$out/test_${name}.txt"
done
mkdir "$out/code"
for file in crvfastsim/veto/bounded_sipm.py crvfastsim/veto/bounded_response.py \
            crvfastsim/veto/saved_qualified_response.py crvfastsim/veto/saved_qualified_report.py \
            tests/test_veto_bounded_sipm.py tests/test_veto_bounded_response.py \
            tests/test_veto_saved_qualified_response.py tests/test_veto_saved_qualified_report.py \
            tests/test_veto_saved_response_admission.py \
            docs/veto-bounded-sipm.md docs/veto-bounded-response.md \
            scripts/verify_veto_saved_qualified_response.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
printf '%s\n' 'Saved response only, no optical/response regeneration or full campaign admission.' > "$out/scope.txt"
