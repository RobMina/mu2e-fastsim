#!/bin/bash
# Inspect a finished pilot only; never run transport or rebuild its runtime.
set -eo pipefail
: "${CALO_CODE:?immutable audit snapshot}"
: "${CALO_OUT:?fresh audit output directory}"
: "${CALO_PILOT:?finished pilot directory}"
: "${CALO_AREA:?existing runtime}"
: "${CALO_RUNNER:?existing chain runner}"
: "${CALO_PYTHON_IMAGE:?existing analysis image}"
: "${CALO_BINDS:?same-path payload binds}"
: "${CVMFSEXEC_DIST:?prepared cvmfsexec}"
module load apptainer
test -s "$CALO_PILOT/calo_library_ce.art"
grep -q '^completed_utc=' "$CALO_PILOT/receipt.txt"
mkdir "$CALO_OUT"
cd "$CALO_OUT"
cp "$CALO_CODE/calo/fcl/status_audit.fcl" status.fcl
cat > chain.sh <<CHAIN
set -eo pipefail
export TMPDIR=\$(dirname "\$OFFLINE_CACHE_PATH")/tmp
export SQLITE_TMPDIR=\$TMPDIR
mkdir -p "\$TMPDIR"
fhicl-dump -c status.fcl > status.expanded.fcl
"\$OFFLINE_TIME" -v -o status.art.time mu2e -c status.expanded.fcl -s "$CALO_PILOT/calo_library_ce.art" > status.log 2>&1
CHAIN
export OFFLINE_BINDS="$CALO_BINDS" OFFLINE_CHAIN="$PWD/chain.sh" OFFLINE_LOG_PREFIX=status
bash "$CALO_RUNNER" "$CALO_AREA" "$PWD/status.fcl" > runner.log 2>&1
apptainer exec --cleanenv -B "$CALO_BINDS" --env "PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1" \
  "$CALO_PYTHON_IMAGE" python3 "$CALO_CODE/calo/scripts/audit_status.py" "$CALO_PILOT/calo_library_ce.art" \
  "$PWD/status.log" --expected-events 2000 --out "$PWD/status.json"
printf 'job=%s\ncompleted_utc=%s\n' "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > receipt.txt
