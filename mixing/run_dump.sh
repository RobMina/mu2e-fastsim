#!/bin/bash
# Extract staged donor frames with existing plugins; no simulation or build.
set -eo pipefail
: "${OFFLINE_CHAIN_RUNNER:?runner with OFFLINE_CHAIN support}"
: "${OFFLINE_AREA:?already-built Offline area with a released backing}"
: "${MIXING_CRV_PLUGIN_DIR:?existing CRV dumper library directory}"
: "${MIXING_TRK_PLUGIN_DIR:?existing tracker dumper library directory}"
: "${MIXING_OUT:?new output directory}"
[[ $# -gt 0 ]] || { echo 'usage: run_dump.sh DONOR.art ...' >&2; exit 2; }
module load apptainer
HERE=$(cd "$(dirname "$0")" && pwd -P)
mkdir -p "$MIXING_OUT"
cd "$MIXING_OUT"
CHAIN="$PWD/chain.sh"
{
  printf 'set -eo pipefail\n'
  printf 'export LD_LIBRARY_PATH=%q:%q:"$LD_LIBRARY_PATH"\n' "$MIXING_TRK_PLUGIN_DIR" "$MIXING_CRV_PLUGIN_DIR"
  printf 'export CET_PLUGIN_PATH=%q:%q:"$CET_PLUGIN_PATH"\n' "$MIXING_TRK_PLUGIN_DIR" "$MIXING_CRV_PLUGIN_DIR"
  printf 'export TMPDIR="$(dirname "$OFFLINE_CACHE_PATH")/tmp"\nmkdir -p "$TMPDIR"\nexport SQLITE_TMPDIR="$TMPDIR"\n'
  for input in "$@"; do
    name=$(basename "$input" .art)
    printf 'config_dumper -P %q > %q\n' "$input" "$name.config.txt"
    printf 'product_sizes_dumper %q > %q\n' "$input" "$name.products.txt"
    printf '"$OFFLINE_TIME" -v -o %q mu2e -c %q -s %q -n %q -T %q > %q 2>&1\n' \
      "$name.time" "$HERE/fcl/dump_frames_legacy.fcl" "$input" "${MIXING_EVENTS:-100}" "$name.root" "$name.log"
  done
} > "$CHAIN"
export OFFLINE_CHAIN="$CHAIN" OFFLINE_LOG_PREFIX=dump
bash "$OFFLINE_CHAIN_RUNNER" "$OFFLINE_AREA" "$HERE/fcl/dump_frames_legacy.fcl" "$@"
