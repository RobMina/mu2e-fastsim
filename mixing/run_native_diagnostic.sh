#!/bin/bash
# One-frame native overlay exercises stock digitization, not physical pileup rates.
set -eo pipefail
: "${OFFLINE_CHAIN_RUNNER:?runner with OFFLINE_CHAIN support}"
: "${OFFLINE_AREA:?already-built Offline area with a released backing}"
: "${MIXING_CRV_PLUGIN_DIR:?existing CRV plugin directory}"
: "${MIXING_TRK_PLUGIN_DIR:?existing tracker plugin directory}"
: "${MIXING_OUT:?new output directory}"
[[ $# == 2 ]] || { echo 'usage: run_native_diagnostic.sh CE.art DONOR.art' >&2; exit 2; }
module load apptainer
HERE=$(cd "$(dirname "$0")" && pwd -P)
SIGNAL=$(readlink -f "$1"); DONOR=$(readlink -f "$2")
mkdir -p "$MIXING_OUT"
cd "$MIXING_OUT"
printf '#include "one_frame_diagnostic.fcl"\nphysics.filters.donor.fileNames: ["%s"]\n' "$DONOR" > job.fcl
CHAIN="$PWD/chain.sh"
{
  printf 'set -eo pipefail\n'
  printf 'export FHICL_FILE_PATH=%q:"$FHICL_FILE_PATH"\n' "$HERE/fcl"
  printf 'export LD_LIBRARY_PATH=%q:%q:"$LD_LIBRARY_PATH"\n' "$MIXING_TRK_PLUGIN_DIR" "$MIXING_CRV_PLUGIN_DIR"
  printf 'export CET_PLUGIN_PATH=%q:%q:"$CET_PLUGIN_PATH"\n' "$MIXING_TRK_PLUGIN_DIR" "$MIXING_CRV_PLUGIN_DIR"
  printf 'export TMPDIR="$(dirname "$OFFLINE_CACHE_PATH")/tmp"\nmkdir -p "$TMPDIR"\nexport SQLITE_TMPDIR="$TMPDIR"\n'
  printf 'fhicl-dump -c job.fcl > expanded.fcl\n'
  printf '"$OFFLINE_TIME" -v -o art.time mu2e -c job.fcl -s %q -n %q -T overlay.root > art.log 2>&1\n' "$SIGNAL" "${MIXING_EVENTS:-10}"
} > "$CHAIN"
export OFFLINE_CHAIN="$CHAIN" OFFLINE_LOG_PREFIX=overlay OFFLINE_FCL_PATH="$HERE/fcl"
export OFFLINE_BINDS="${OFFLINE_BINDS:+$OFFLINE_BINDS,}$HERE:$HERE"
bash "$OFFLINE_CHAIN_RUNNER" "$OFFLINE_AREA" "$PWD/job.fcl" "$SIGNAL" "$DONOR"
