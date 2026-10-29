#!/usr/bin/env bash
# Verify saved keyed photons and run isolated tests; no new response draw.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 EXISTING_SMOKE NEW_OUTPUT" >&2
  exit 2
fi
smoke=$1
out=$2
mkdir "$out"
python3 -m crvfastsim.veto.keyed_optical_smoke --verify "$smoke" --out "$out/verified.json"
for name in genassoc_campaign veto_cat_primary_audit veto_census_policy veto_library_weights \
            veto_step_timing_audit veto_optical_timing_probe veto_optical_timing_report \
            veto_replay_sipm_control veto_replay_waveform_control veto_saved_downstream_report \
            veto_saved_photon_downstream veto_keyed_optics veto_keyed_optical_smoke veto_replay_timing veto_replay; do
  python3 -m pytest -q "tests/test_${name}.py" > "$out/test_${name}.txt" 2>&1
  cat "$out/test_${name}.txt"
done
mkdir "$out/code"
for file in crvfastsim/veto/keyed_optics.py crvfastsim/veto/keyed_optical_smoke.py \
            tests/test_veto_keyed_optics.py tests/test_veto_keyed_optical_smoke.py \
            docs/veto-keyed-optics.md docs/veto-keyed-optical-smoke.md scripts/verify_veto_keyed_smoke.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
