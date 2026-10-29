#!/usr/bin/env bash
# Verify saved resource planning and run isolated tests; no new detector response.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 EXISTING_PLAN NEW_OUTPUT" >&2
  exit 2
fi
plan=$1
out=$2
mkdir "$out"
python3 -m crvfastsim.veto.retained_plan_report "$plan" --out "$out/verified.json"
for name in genassoc_campaign veto_cat_primary_audit veto_census_policy veto_library_weights \
            veto_step_timing_audit veto_optical_timing_probe veto_optical_timing_report \
            veto_replay_sipm_control veto_replay_waveform_control veto_saved_downstream_report \
            veto_saved_photon_downstream veto_keyed_optics veto_keyed_optical_smoke \
            veto_frame_registry veto_retained_plan veto_retained_plan_report veto_qualified_response \
            veto_replay_timing veto_replay; do
  python3 -m pytest -q -s "tests/test_${name}.py" > "$out/test_${name}.txt" 2>&1
  cat "$out/test_${name}.txt"
done
mkdir "$out/code"
for file in crvfastsim/veto/frame_registry.py crvfastsim/veto/retained_plan.py \
            crvfastsim/veto/retained_plan_report.py crvfastsim/veto/qualified_response.py \
            tests/test_veto_frame_registry.py tests/test_veto_retained_plan.py \
            tests/test_veto_retained_plan_report.py tests/test_veto_qualified_response.py \
            docs/veto-retained-plan.md scripts/verify_veto_retained_plan.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
