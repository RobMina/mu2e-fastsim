#!/bin/bash
# Run one bounded unfiltered pilot with an existing runtime and audited expansion.
set -eo pipefail
: "${CALO_CODE:?immutable snapshot}"
: "${CALO_OUT:?fresh output directory}"
: "${CALO_AREA:?already-built runtime}"
: "${CALO_RUNNER:?existing Offline chain runner}"
: "${CALO_FCL:?audited expanded FCL}"
: "${CALO_FCL_SHA256:?audited expanded FCL checksum}"
: "${CALO_POOL:?bounded upstream stop pool}"
: "${CALO_POOL_SHA256:?upstream subset checksum}"
: "${CALO_AREA_SHA256:?runtime checksum}"
: "${CALO_BINDS:?same-path payload binds}"
: "${CVMFSEXEC_DIST:?prepared cvmfsexec}"
test "${SLURM_CPUS_PER_TASK:-1}" = 1
module load apptainer
echo "$CALO_FCL_SHA256  $CALO_FCL" | sha256sum -c -
echo "$CALO_POOL_SHA256  $CALO_POOL" | sha256sum -c -
echo "$CALO_AREA_SHA256  $CALO_AREA" | sha256sum -c -
mkdir "$CALO_OUT"
cd "$CALO_OUT"
printf 'job=%s\nhost=%s\nrequested_events=2000\nstart_utc=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" > receipt.txt
cp "$CALO_FCL" approved.expanded.fcl
sha256sum "$CALO_FCL" "$CALO_POOL" "$CALO_AREA" "$CALO_RUNNER" > inputs.sha256
cat > chain.sh <<'CHAIN'
set -eo pipefail
export TMPDIR=$(dirname "$OFFLINE_CACHE_PATH")/tmp
export SQLITE_TMPDIR=$TMPDIR
mkdir -p "$TMPDIR"
mu2e --version > runtime.version.txt
fhicl-dump -c approved.expanded.fcl > executed.expanded.fcl
"$OFFLINE_TIME" -v -o pilot.art.time mu2e -c executed.expanded.fcl -n 2000 > pilot.log 2>&1
grep -E '^Geom: ' pilot.log > geometry.resolved.txt
test -s geometry.resolved.txt
sha256sum geometry.resolved.txt executed.expanded.fcl > provenance.sha256
config_dumper -P calo_library_ce.art > output.config.txt
"$OFFLINE_TIME" -v -o products.art.time mu2e -c products.fcl -n 1 -s calo_library_ce.art > products.log 2>&1
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> receipt.txt
CHAIN
cp "$CALO_CODE/calo/fcl/products.fcl" products.fcl
export OFFLINE_BINDS="$CALO_BINDS" OFFLINE_CHAIN="$PWD/chain.sh" OFFLINE_LOG_PREFIX=pilot
bash "$CALO_RUNNER" "$CALO_AREA" "$PWD/approved.expanded.fcl" > runner.log 2>&1
