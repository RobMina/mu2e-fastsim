#!/bin/bash
# Isolate each statically selected file without changing the historical selection list.
set -euo pipefail
: "${CRV_TIER1B_ROOT:?}" "${CRV_TIER1B_CODE:?}" "${CRV_DEFAULT_SELECTION:?}"
failed=0
while IFS= read -r test; do
 echo "== $test"
 bash "$CRV_TIER1B_CODE/crv/scripts/device_tier1b_tests_uva.sh" "$test" || failed=1
done < "$CRV_DEFAULT_SELECTION"
exit "$failed"
