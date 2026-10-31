#!/bin/bash
# run_fast_cpu.sh G4DIR OUTDIR N [extra run args] : G4 table (once), field check, fast transport of the first N muons, comparison (GPVM CPU)
G4=$1; OUT=$2; N=${3:-20000}; shift 3
HERE=$(cd "$(dirname "$0")/../.." && pwd)
VENV=${VENV:-${FASTSIMKIT_EAF_VENV:?set VENV or FASTSIMKIT_EAF_VENV to a Python/JAX venv (site.env)}}
export PYTHONPATH=$HERE/core:$HERE/beam JAX_PLATFORMS=cpu
mkdir -p "$OUT"
PY=$VENV/bin/python
[ -f "$G4/g4ref.npz" ] || $PY -m beamfastsim g4table --nts "$G4"/nts.*.root --out "$G4/g4ref.npz" || exit 2
$PY -m beamfastsim fieldcheck || exit 3
$PY -m beamfastsim run --g4 "$G4/g4ref.npz" --out "$OUT/fast.npz" --n "$N" "$@" || exit 4
$PY -m beamfastsim compare --g4 "$G4/g4ref.npz" --fast "$OUT/fast.npz" --out "$OUT" || exit 5
