#!/bin/bash
# Check source bindings in a new temporary review tree without native execution.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Slurm allocation required' >&2; exit 2; }
[[ $# -eq 3 ]] || { echo 'usage: tsrecordproposal_checks.sh SNAPSHOT ROOT PINNED_OFFLINE' >&2; exit 2; }
SNAPSHOT=$1
ROOT=$2
OFFLINE=$3
module load apptainer
: "${FASTSIM_CORE_ROOT:?set FASTSIM_CORE_ROOT to the existing pinned core source}"
: "${FASTSIM_IMAGE:?set FASTSIM_IMAGE to the existing analysis image}"
: "${FASTSIM_BIND_ROOT:?set FASTSIM_BIND_ROOT to the shared input/output ancestor}"
export APPTAINERENV_PYTHONPATH="$SNAPSHOT/beam:$FASTSIM_CORE_ROOT"
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
IMAGE=$FASTSIM_IMAGE
apptainer exec --cleanenv -B "$FASTSIM_BIND_ROOT" "$IMAGE" \
  python3 -m pytest -q -p no:cacheprovider "$SNAPSHOT/beam/tests/test_tsrecordproposal.py" \
  "$SNAPSHOT/beam/tests/test_tsrecordsource.py" "$SNAPSHOT/beam/tests/test_tsrecordledger.py" \
  "$SNAPSHOT/beam/tests/test_tsrecordsourceledger.py" "$SNAPSHOT/beam/tests/test_tsrecordreadback.py"
apptainer exec --cleanenv -B "$FASTSIM_BIND_ROOT" "$IMAGE" \
  python3 -m beamfastsim.tsrecordproposal --source-root "$OFFLINE" \
  --proposal-root "$SNAPSHOT/beam/native" --destination "$ROOT/review_tree_${SLURM_JOB_ID}" \
  --out "$ROOT/target_proposal_${SLURM_JOB_ID}.json"
apptainer exec --cleanenv -B "$FASTSIM_BIND_ROOT" "$IMAGE" \
  python3 -m pytest -q -p no:cacheprovider "$SNAPSHOT/beam/tests"/test_ts*.py
