#!/usr/bin/env bash
# Saved-photon downstream evidence only: no new optical or detector-response draws.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 EXISTING_DOWNSTREAM NEW_OUTPUT" >&2
  exit 2
fi
prior=$1
out=$2
mkdir "$out"
python3 -m crvfastsim.veto.saved_downstream_report "$prior" --out "$out/report"
for name in genassoc_campaign veto_cat_primary_audit veto_census_policy veto_library_weights \
            veto_step_timing_audit veto_optical_timing_probe veto_optical_timing_report \
            veto_replay_sipm_control veto_replay_waveform_control veto_saved_photon_downstream \
            veto_saved_downstream_report veto_replay_timing veto_replay; do
  python3 -m pytest -q "tests/test_${name}.py" > "$out/test_${name}.txt" 2>&1
  cat "$out/test_${name}.txt"
done
mkdir "$out/code"
for file in crvfastsim/veto/saved_photon_downstream.py crvfastsim/veto/saved_downstream_report.py \
            crvfastsim/veto/replay_sipm_control.py crvfastsim/veto/replay_waveform_control.py \
            tests/test_veto_saved_photon_downstream.py tests/test_veto_saved_downstream_report.py \
            tests/test_veto_replay_sipm_control.py tests/test_veto_replay_waveform_control.py \
            docs/veto-saved-photon-downstream.md scripts/verify_veto_saved_downstream.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
