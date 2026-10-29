#!/usr/bin/env bash
# Verify retained step ledgers and run isolated tests, without response or source re-extraction.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 EXISTING_AUDIT OFFLINE_DIRECTORY NEW_OUTPUT" >&2
  exit 2
fi
audit=$1
offline=$2
out=$3
mkdir "$out"
python3 -m crvfastsim.veto.step_timing_audit --verify "$audit" \
  --counters "$offline/Mu2eG4/geom/crv_counters_run1a_v01.txt" \
  --stock-source "$offline/CRVResponse/src/CrvPhotonGenerator_module.cc" \
  --out "$out/verified.json"
for name in genassoc_campaign veto_cat_primary_audit veto_census_policy veto_library_weights \
            veto_step_timing_audit veto_replay_timing veto_replay; do
  python3 -m pytest -q "tests/test_${name}.py" > "$out/test_${name}.txt" 2>&1
  cat "$out/test_${name}.txt"
done
mkdir "$out/code"
for file in crvfastsim/veto/step_timing_audit.py tests/test_veto_step_timing_audit.py \
            docs/veto-step-timing-audit.md scripts/verify_veto_step_timing.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
