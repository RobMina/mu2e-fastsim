#!/bin/bash
# Isolated change-selected tests in the existing CPU image.
set -euo pipefail
: "${CRV_TIER1B_ROOT:?}" "${CRV_TIER1B_CODE:?}"
failed=0
while IFS= read -r test; do
 echo "== $test"
 bash "$CRV_TIER1B_CODE/crv/scripts/device_tier1b_tests_uva.sh" "$test" || failed=1
done < "$CRV_TIER1B_ROOT/selected_tests.txt"
exit "$failed"
