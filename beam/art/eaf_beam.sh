#!/bin/bash
# eaf_beam.sh OUTDIR [N] : beam fast transport on an EAF A100 slice (run in the EAF session via core/eaf/eafhub.py run).
# Inputs read in place from /exp (G4 table) and CVMFS (field map, geometry); writes env.txt, bench.json, fast.npz, compare.
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
D=/exp/mu2e/data/users/ram2aq
VENV=${VENV:-$D/crv_fastsim/venv_eaf}
O=$1; N=${2:-105350}
PY=$VENV/bin/python
export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/beam XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$O" || exit 2
{ hostname; git -C "$M2FS_CODE" log -1 --format="umbrella %h"; nvidia-smi -L 2>/dev/null
  "$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"; } | tee "$O/env.txt"
grep -q -i cuda "$O/env.txt" || { echo "JAX sees no GPU"; exit 5; }
"$PY" -m beamfastsim.bench --n ${BENCH_N:-200000} --reps 2 --chunk 400 > "$O/bench.json" 2>&1 || exit 6
"$PY" -m beamfastsim run --g4 "$D/beam_fastsim/g4ref_full/g4ref.npz" --out "$O/fast.npz" --n "$N" --batch 131072 \
  --chunk 400 > "$O/run.log" 2>&1 || exit 7
"$PY" -m beamfastsim compare --g4 "$D/beam_fastsim/g4ref_full/g4ref.npz" --fast "$O/fast.npz" --out "$O" > "$O/compare.log" 2>&1 || exit 8
echo done
