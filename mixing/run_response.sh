#!/bin/bash
# Run saved-digi reco or four-stream response with an existing Offline runtime.
set -eo pipefail
: "${OFFLINE_CHAIN_RUNNER:?runner with OFFLINE_CHAIN support}"
: "${OFFLINE_AREA:?already-built Offline area}"
: "${MIXING_CRV_PLUGIN_DIR:?existing CRV plugin directory}"
: "${MIXING_TRK_PLUGIN_DIR:?existing tracker plugin directory}"
: "${MIXING_OUT:?new output directory}"
[[ $# == 2 || $# == 6 ]] || { echo 'usage: run_response.sh CONFIG.fcl SIGNAL.art [STOP.art ELE.art MU.art NEUTRAL.art]' >&2; exit 2; }
module load apptainer
HERE=$(cd "$(dirname "$0")" && pwd -P)
CONFIG=$(readlink -f "$1"); shift
INPUTS=()
for input in "$@"; do INPUTS+=("$(readlink -f "$input")"); done
mkdir -p "$MIXING_OUT"
cd "$MIXING_OUT"
printf '#include "%s"\n' "$(basename "$CONFIG")" > job.fcl
if [[ -n ${MIXING_SEED:-} ]]; then
  [[ $MIXING_SEED =~ ^[0-9]+$ ]] || { echo 'MIXING_SEED must be an integer' >&2; exit 2; }
  printf 'services.SeedService.baseSeed: %s\n' "$MIXING_SEED" >> job.fcl
fi
if [[ -n ${MIXING_DIGI_TABLE:-} ]]; then
  printf 'physics.producers.tableSD.tableFile: "%s"\n' "$(readlink -f "$MIXING_DIGI_TABLE")" >> job.fcl
fi
if [[ ${#INPUTS[@]} == 5 ]]; then
  labels=(MuStopPileupMixer EleBeamFlashMixer MuBeamFlashMixer NeutralsFlashMixer)
  for i in 0 1 2 3; do
    printf 'physics.filters.%s.fileNames: ["%s"]\n' "${labels[$i]}" "${INPUTS[$((i+1))]}" >> job.fcl
  done
fi
CHAIN="$PWD/chain.sh"
{
  printf 'set -eo pipefail\n'
  printf 'export FHICL_FILE_PATH=%q:%q:"$FHICL_FILE_PATH"\n' "$(dirname "$CONFIG")" "$HERE/fcl"
  printf 'export LD_LIBRARY_PATH=%q:%q:"$LD_LIBRARY_PATH"\n' "$MIXING_TRK_PLUGIN_DIR" "$MIXING_CRV_PLUGIN_DIR"
  printf 'export CET_PLUGIN_PATH=%q:%q:"$CET_PLUGIN_PATH"\n' "$MIXING_TRK_PLUGIN_DIR" "$MIXING_CRV_PLUGIN_DIR"
  printf 'export TMPDIR="$(dirname "$OFFLINE_CACHE_PATH")/tmp"\nmkdir -p "$TMPDIR"\nexport SQLITE_TMPDIR="$TMPDIR"\n'
  printf 'fhicl-dump -c job.fcl > expanded.fcl\n'
  printf 'config_dumper -P %q > input.config.txt\n' "${INPUTS[0]}"
  if [[ ${MIXING_EXPORT_CONDITIONS:-0} == 1 ]]; then
    for table in TrkPreampStraw TrkAlignStrawSim; do
      printf 'dbTool print-run --purpose Sim_best --version v1_3 --run 1430:1253 --table %q --content > %q\n' "$table" "db_$table.csv"
    done
    printf 'cp "$MUSE_WORK_DIR/backing/Offline/GlobalConstantsService/data/globalConstants_01.txt" globalConstants.txt\n'
  fi
  if [[ ${MIXING_EXPORT_CRV_CONDITIONS:-0} == 1 ]]; then
    printf 'dbTool print-run --purpose Sim_best --version v1_3 --run 1430:1253 --table CRVPhoton --content > db_CRVPhoton.csv\n'
  fi
  printf '"$OFFLINE_TIME" -v -o art.time mu2e -c job.fcl -s %q -n %q -T response.root > art.log 2>&1\n' "${INPUTS[0]}" "${MIXING_EVENTS:-10}"
} > "$CHAIN"
export OFFLINE_CHAIN="$CHAIN" OFFLINE_LOG_PREFIX=response OFFLINE_FCL_PATH="$(dirname "$CONFIG"):$HERE/fcl"
export OFFLINE_BINDS="${OFFLINE_BINDS:+$OFFLINE_BINDS,}$HERE:$HERE,$(dirname "$CONFIG"):$(dirname "$CONFIG")"
bash "$OFFLINE_CHAIN_RUNNER" "$OFFLINE_AREA" "$PWD/job.fcl" "${INPUTS[@]}"
