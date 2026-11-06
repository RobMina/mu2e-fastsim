#!/bin/bash
# Check an opt-in native preparation against a saved full-development output.
set -eo pipefail
[[ $# == 8 ]] || { echo 'usage: run_preparation_uva.sh NEW_OUT REFERENCE_OUT EXPECTED_EVENTS SIGNAL.art STOP.art ELE.art MU.art NEUTRAL.art' >&2; exit 2; }
: "${SLURM_JOB_ID:?run in a Slurm CPU allocation}"
: "${MIXING_SIF:?existing Python analysis image}"
: "${MIXING_DATA_ROOT:?staged data and runtime root}"
HERE=$(cd "$(dirname "$0")" && pwd -P)
CODE=$(dirname "$HERE")
OUT=$1; REF=$(readlink -f "$2"); EXPECTED=$3; shift 3
[[ $EXPECTED =~ ^[1-9][0-9]*$ ]] || { echo 'EXPECTED_EVENTS must be positive' >&2; exit 2; }
mkdir "$OUT"
OUT=$(readlink -f "$OUT")
printf 'host=%s\njob=%s\ncode=%s\nreference=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$CODE" "$REF" > "$OUT/run.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONFIG="$HERE/fcl/four_stream_tracker_crv_inputs.fcl"
MIXING_EVENTS=10 MIXING_OUT="$OUT/smoke" bash "$HERE/run_response.sh" "$CONFIG" "$@"
MIXING_EVENTS=-1 MIXING_EXPORT_CONDITIONS=1 MIXING_OUT="$OUT/full" bash "$HERE/run_response.sh" "$CONFIG" "$@"
module load apptainer
BINDS="$MIXING_DATA_ROOT:$MIXING_DATA_ROOT,$CODE:$CODE,$REF:$REF,$OUT:$OUT"
apptainer exec --cleanenv -B "$BINDS" \
  --env "PYTHONPATH=$CODE,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1" "$MIXING_SIF" \
  python3 "$CODE/container/offline/compare_root.py" "$REF/response.root" "$OUT/full/response.root" --output "$OUT/comparison.json"
apptainer exec --cleanenv -B "$BINDS" \
  --env "PYTHONPATH=$CODE,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1" "$MIXING_SIF" \
  python3 -m mixing.preparation_report --reference "$REF" --candidate "$OUT/full" \
  --comparison "$OUT/comparison.json" --expected-events "$EXPECTED" --out "$OUT/report.json"
