#!/usr/bin/env bash
# Verify existing streaming pilot evidence without regenerating optics or detector response.
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "usage: $0 EXISTING_PILOT PLAN_SHA CACHE_SHA RESPONSE_SHA NEW_OUTPUT" >&2
  exit 2
fi
existing=$1
plan_sha=$2
cache_sha=$3
response_sha=$4
out=$5
mkdir "$out"
python3 -m crvfastsim.veto.streaming_pilot_report "$existing" --expected-plan-sha256 "$plan_sha" \
  --expected-cache-sha256 "$cache_sha" --expected-response-sha256 "$response_sha" \
  --out "$out/verified.json" --compact > "$out/compact.json"
python3 -m mu2efastsim.testselect --why > "$out/selection.txt"
python3 -m mu2efastsim.testselect --run --isolate -- -q -s > "$out/selected_tests.txt" 2>&1
for name in veto_replay_timing veto_replay; do
  python3 -m pytest -p no:cacheprovider -q "tests/test_${name}.py" > "$out/test_${name}.txt" 2>&1
done
python3 --version > "$out/python_version.txt"
mkdir "$out/code"
for file in crvfastsim/veto/streaming_optics.py crvfastsim/veto/streaming_pilot.py \
            crvfastsim/veto/streaming_pilot_report.py tests/test_veto_streaming_optics.py \
            tests/test_veto_streaming_pilot.py tests/test_veto_streaming_pilot_report.py \
            docs/veto-streaming-pilot.md scripts/verify_veto_streaming_pilot.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
printf '%s\n' 'Existing bounded pilot only; inherited raw/stock admission reads, no physics regeneration or full campaign admission.' > "$out/scope.txt"
