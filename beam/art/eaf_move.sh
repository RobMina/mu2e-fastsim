#!/bin/bash
# eaf_move.sh OUTDIR : geometry-move A/B, fast arm on an EAF A100 slice. Nominal and move_extras12 on the same
# MDC2025ac muons (warm-pass throughput included), each compared with its own G4 arm (g4ref_full, g4ref_move_full).
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
D=/exp/mu2e/data/users/ram2aq/beam_fastsim
VENV=${VENV:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf}
O=$1; PY=$VENV/bin/python; MV=$M2FS_CODE/beam/geom/move_extras12.txt
export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/beam XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$O" || exit 2
{ hostname; git -C "$M2FS_CODE" log -1 --format="umbrella %h"; nvidia-smi -L; "$PY" -c "import jax; print(jax.devices())"; } | tee "$O/env.txt"
grep -q -i cuda "$O/env.txt" || { echo "JAX sees no GPU"; exit 5; }
"$PY" -m beamfastsim point --g4 $D/g4ref_full/g4ref.npz --out "$O/nominal" --repeat 2 > "$O/nominal.log" 2>&1 || exit 6
"$PY" -m beamfastsim point --g4 $D/g4ref_full/g4ref.npz --out "$O/move" --geom "$MV" --repeat 2 > "$O/move.log" 2>&1 || exit 7
[ -f $D/g4ref_move_full/g4ref.npz ] || "$PY" -m beamfastsim g4table --nts $D/g4ref_move_full/nts.*.root --geom "$MV" --out $D/g4ref_move_full/g4ref.npz || exit 8
for arm in nominal move; do
  G=$D/g4ref_full/g4ref.npz; GM=""; [ $arm = move ] && { G=$D/g4ref_move_full/g4ref.npz; GM="--geom $MV"; }
  "$PY" -m beamfastsim run --g4 $G --out "$O/$arm/fast_cmp.npz" $GM --batch 131072 --chunk 400 > "$O/$arm/run_cmp.log" 2>&1 || exit 9
  "$PY" -m beamfastsim compare --g4 $G --fast "$O/$arm/fast_cmp.npz" $GM --out "$O/$arm/compare" > "$O/$arm/compare.log" 2>&1 || exit 10
  rm -f "$O/$arm/fast_cmp.npz"
done
echo done
