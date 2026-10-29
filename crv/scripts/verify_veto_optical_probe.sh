#!/usr/bin/env bash
# Verify saved photons and run isolated tests; no new optical or downstream response draw.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 EXISTING_PROBE NEW_OUTPUT" >&2
  exit 2
fi
probe=$1
out=$2
mkdir "$out"
python3 -m crvfastsim.veto.optical_timing_report "$probe" --out "$out/verified.json"
for name in genassoc_campaign veto_cat_primary_audit veto_census_policy veto_library_weights \
            veto_step_timing_audit veto_optical_timing_probe veto_optical_timing_report veto_replay_timing veto_replay; do
  python3 -m pytest -q "tests/test_${name}.py" > "$out/test_${name}.txt" 2>&1
  cat "$out/test_${name}.txt"
done
mkdir "$out/code"
for file in crvfastsim/veto/optical_timing_probe.py crvfastsim/veto/optical_timing_report.py \
            tests/test_veto_optical_timing_probe.py tests/test_veto_optical_timing_report.py \
            docs/veto-optical-timing-probe.md scripts/verify_veto_optical_probe.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
