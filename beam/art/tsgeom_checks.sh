#!/bin/bash
# Test the upstream tools and audit existing exports, without invoking simulation.
set -eo pipefail
[[ $# -eq 3 ]] || { echo 'usage: tsgeom_checks.sh CODE_ROOT EXPORT_ROOT AUDIT_SUFFIX' >&2; exit 2; }
CODE=$1
ROOT=$2
SUFFIX=$3
python3 -m pytest -q -p no:cacheprovider "$CODE"/beam/tests/test_ts*.py "$CODE/tests"
bash "$CODE/beam/art/tsgdml_audit.sh" "$ROOT" "$SUFFIX"
bash "$CODE/beam/art/tsgeom_audit.sh" "$ROOT" "$SUFFIX"
bash "$CODE/beam/art/tschord_audit.sh" "$ROOT" "$SUFFIX"
bash "$CODE/beam/art/tsarc_audit.sh" "$ROOT" "$SUFFIX"
