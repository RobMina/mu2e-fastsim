#!/usr/bin/env bash
# Test diagnostic prerequisites without reading retained physics inputs or generating response.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: $0 NEW_OUTPUT" >&2
  exit 2
fi
out=$1
mkdir "$out"
python3 -m mu2efastsim.testselect --why > "$out/selection.txt"
python3 -m mu2efastsim.testselect --run --isolate -- -q -s > "$out/selected_tests.txt" 2>&1
cat "$out/selected_tests.txt"
for name in veto_replay_timing veto_replay; do
  python3 -m pytest -p no:cacheprovider -q "tests/test_${name}.py" > "$out/test_${name}.txt" 2>&1
  cat "$out/test_${name}.txt"
done
python3 --version > "$out/python_version.txt"
mkdir "$out/code"
for file in crvfastsim/veto/indexed_replay.py crvfastsim/veto/batch_artifacts.py \
            crvfastsim/veto/diagnostic_fit.py crvfastsim/veto/diagnostic_lm.py \
            crvfastsim/veto/bounded_waveform.py tests/test_veto_indexed_replay.py \
            tests/test_veto_batch_artifacts.py tests/test_veto_diagnostic_fit.py \
            tests/test_veto_bounded_waveform.py docs/veto-indexed-replay.md \
            docs/veto-batch-artifacts.md docs/veto-diagnostic-fit.md \
            scripts/verify_veto_execution_prerequisites.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
sha256sum crvfastsim/reco/pulsefit.py crvfastsim/reco/waveform.py \
  crvfastsim/veto/frame_registry.py crvfastsim/veto/retained_plan.py \
  crvfastsim/veto/retained_plan_report.py crvfastsim/veto/qualified_response.py \
  crvfastsim/veto/keyed_optics.py crvfastsim/veto/keyed_optical_smoke.py \
  crvfastsim/veto/replay_sipm_control.py crvfastsim/veto/replay_waveform_control.py \
  crvfastsim/veto/saved_photon_downstream.py > "$out/frozen_sha256.txt"
printf '%s\n' 'Synthetic prerequisites only. execution_admitted=false. No real response or F.' > "$out/scope.txt"
