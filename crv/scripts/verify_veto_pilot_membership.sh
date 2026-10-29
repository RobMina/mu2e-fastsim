#!/usr/bin/env bash
# Verify existing membership evidence and test the new saved-only verifier without physics regeneration.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 EXISTING_MEMBERSHIP SUMMARY_SHA NEW_OUTPUT" >&2
  exit 2
fi
existing=$1
summary_sha=$2
out=$3
mkdir "$out"
python3 -m crvfastsim.veto.pilot_membership_report "$existing" --expected-summary-sha256 "$summary_sha" \
  --out "$out/verified.json" --compact > "$out/compact.json"
python3 -m pytest -p no:cacheprovider -q tests/test_veto_pilot_membership_report.py > "$out/verifier_tests.txt" 2>&1
python3 --version > "$out/python_version.txt"
mkdir "$out/code"
for file in crvfastsim/veto/pilot_coincidence.py crvfastsim/veto/pilot_coverage.py \
            crvfastsim/veto/pilot_membership.py crvfastsim/veto/pilot_membership_report.py \
            tests/test_veto_pilot_coincidence.py tests/test_veto_pilot_coverage.py \
            tests/test_veto_pilot_membership.py tests/test_veto_pilot_membership_report.py \
            docs/veto-pilot-membership.md scripts/verify_veto_pilot_membership.sh; do
  sha256sum "$file" >> "$out/code_sha256.txt"
  cp "$file" "$out/code/"
done
printf '%s\n' 'Existing saved membership only; no finder/upstream physics/raw-admission regeneration. New verifier tests only; prior adapter/coverage/runner test logs are separate retained evidence.' > "$out/scope.txt"
