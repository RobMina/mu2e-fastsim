#!/bin/bash
# Extract complete joint source pools once with the existing legacy dumpers.
set -eo pipefail
[[ $# == 5 ]] || { echo 'usage: run_frame_cache.sh SIGNAL STOP ELE MU NEUTRAL' >&2; exit 2; }
CODE=$(cd "$(dirname "$0")/.." && pwd -P)
OUT=$(readlink -m "${MIXING_OUT:?set a new cache output directory}")
mkdir "$OUT"
export OFFLINE_CHAIN=$OUT/chain.sh
export OFFLINE_BINDS="${OFFLINE_BINDS:+$OFFLINE_BINDS,}$CODE:$CODE,$OUT:$OUT"
{
  printf 'set -eo pipefail\n'
  printf 'export LD_LIBRARY_PATH=%q:%q:"$LD_LIBRARY_PATH"\n' "${MIXING_TRK_PLUGIN_DIR:?}" "${MIXING_CRV_PLUGIN_DIR:?}"
  printf 'export CET_PLUGIN_PATH=%q:%q:"$CET_PLUGIN_PATH"\n' "$MIXING_TRK_PLUGIN_DIR" "$MIXING_CRV_PLUGIN_DIR"
  printf 'export TMPDIR="$(dirname "$OFFLINE_CACHE_PATH")/tmp"\nmkdir -p "$TMPDIR"\nexport SQLITE_TMPDIR="$TMPDIR"\n'
  printf 'dbTool print-run --purpose Sim_best --version v1_3 --run 1430:1253 --table SimEfficiencies2 --content > db_SimEfficiencies2.csv\n'
  names=(signal MuStopPileup EleBeamFlash MuBeamFlash NeutralsFlash)
  i=0
  for input in "$@"; do
    name=${names[$i]}
    printf '"$OFFLINE_TIME" -v -o %q mu2e -c %q -s %q -n -1 -T %q > %q 2>&1\n' \
      "$name.time" "$CODE/mixing/fcl/dump_frames_legacy.fcl" "$(readlink -f "$input")" "$name.root" "$name.log"
    i=$((i+1))
  done
} > "$OFFLINE_CHAIN"
cd "$OUT"
export OFFLINE_LOG_PREFIX=frame_cache
bash "${OFFLINE_CHAIN_RUNNER:?}" "${OFFLINE_AREA:?}" "$CODE/mixing/fcl/dump_frames_legacy.fcl" "$@"
