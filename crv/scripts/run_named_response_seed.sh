#!/usr/bin/env bash
# One serial traced response replay on saved stock event identities, with all declared yields unchanged.
set -eo pipefail
OUT=${1:?new output directory}
BASE=${2:?named stock and conditions root}
STEPS=${3:?CrvStep table}
CORE=${4:?explicit core checkout}
RUN=${5:?run1a or run2}
BACKEND=${6:?numpy or jax or fused}
SEED=${7:?fresh response seed}
case "$RUN" in
  run1a) RC=run1a_v01; tables=(6000_0) ;;
  run2) RC=run2_v10; tables=(4550_0 1045_1 3040_0 3200_0 6000_1 6000_0 5000_1 6900_1 5700_0 2370_1 2100_1 1550_1) ;;
  *) echo "unsupported run: $RUN" >&2; exit 2 ;;
esac
case "$BACKEND" in
  numpy) backend=() ;;
  jax) backend=(--jax) ;;
  fused) backend=(--jax --fused-response) ;;
  *) echo "unsupported backend: $BACKEND" >&2; exit 2 ;;
esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a nonnegative integer" >&2; exit 2; }
LOOKUP=${CRVFS_LOOKUP_DIR:?explicit optical directory required}
OFFLINE=${CRVFS_VALIDATION_OFFLINE:?explicit Offline source and waveform root required}
mkdir "$OUT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$CORE:$PWD" CRVFS_NO_FETCH=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export CRVFASTSIM_COMPACT_DISPATCH=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false MU2E_SEARCH_PATH="$OFFLINE/.."
if [[ $(uname -s) = Linux ]]; then
  CPU=$(awk '/Cpus_allowed_list/ {split($2,a,/[,-]/); print a[1]}' /proc/self/status)
  [[ "$CPU" =~ ^[0-9]+$ ]] || { echo "cannot determine allowed CPU" >&2; exit 2; }
  launch=(taskset -c "$CPU"); timing=(-v); digest=(sha256sum)
else
  CPU=host_cpu_no_affinity
  launch=(); timing=(-lp); digest=(shasum -a 256)
fi
REF="$BASE/stock_1k_v1/$RUN/crvreco.root"
MAN="$BASE/conditions_v1/$RUN/manifest.json"
{
  printf 'host=%s\nrun=%s\nbackend=%s\nseed=%s\ncpu=%s\n' "$(hostname)" "$RC" "$BACKEND" "$SEED" "$CPU"
  printf 'python='; command -v python3
  python3 --version
  printf 'scope=conditional fixed G4 steps and stock event population; no stock seed replication or V2 certification\n'
} > "$OUT/invocation.txt"
"${digest[@]}" "$STEPS" "$REF" "$MAN" "$BASE/conditions_v1/$RUN/"{calib.csv,photon.csv,counters.txt} > "$OUT/inputs.sha256"
"${digest[@]}" "$LOOKUP/photonMap.root" "$OFFLINE/CRVResponse/data/singlePEWaveform_v3.txt" > "$OUT/optical.sha256"
for table in "${tables[@]}"; do "${digest[@]}" "$LOOKUP/LookupTable_$table" >> "$OUT/optical.sha256"; done
"${digest[@]}" crvfastsim/chain.py crvfastsim/readout.py crvfastsim/runconfig.py crvfastsim/named_conditions.py \
  crvfastsim/geometry.py crvfastsim/tables.py crvfastsim/response/*.py crvfastsim/reco/*.py \
  crvfastsim/analyses/named_response_audit.py crvfastsim/analyses/named_response_assets.py \
  valhist/fasttables.py "$0" > "$OUT/sources.sha256"
python3 -m crvfastsim.analyses.named_response_assets --run-config "$RC" --offline "$OFFLINE" > "$OUT/resolved-assets.before.json"
/usr/bin/time "${timing[@]}" "${launch[@]}" python3 -m valhist.fasttables chain "$STEPS" "$OUT/candidate.root" \
  --run-config "$RC" --conditions-manifest "$MAN" --event-reference "$REF" \
  --geometry-readout --trace-response --seed "$SEED" "${backend[@]}" > "$OUT/produce.log" 2>&1
"${launch[@]}" python3 -m crvfastsim.analyses.named_response_audit \
  --reference "$REF" --candidate "$OUT/candidate.root" --conditions-manifest "$MAN" \
  --run-config "$RC" --outdir "$OUT/audit" > "$OUT/audit.log" 2>&1
"${digest[@]}" -c "$OUT/inputs.sha256" "$OUT/optical.sha256" "$OUT/sources.sha256" > "$OUT/recheck.log"
python3 -m crvfastsim.analyses.named_response_assets --run-config "$RC" --offline "$OFFLINE" > "$OUT/resolved-assets.after.json"
cmp "$OUT/resolved-assets.before.json" "$OUT/resolved-assets.after.json"
printf 'complete\n' > "$OUT/complete.txt"
cat "$OUT/invocation.txt" "$OUT/produce.log" "$OUT/audit/table.md"
