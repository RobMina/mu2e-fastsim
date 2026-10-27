#!/bin/bash
# Keep detailed physics outputs in private node tmpfs; publish only summary diagnostics.
set -eo pipefail
L=${1:?}; O=${2:?}; V=${3:?}; SITE=${4:?}
cd "$L/code/crv"
extra=()
if test "$SITE" = aurora; then extra=(--reco-cpu); fi
if test "${AGENT13_REUSE_CONTEXT:-0}" = 1; then extra+=(--reuse-context); fi
printf 'site=%s cpu_affinity=%s reco_cpu=%s\n' "$SITE" "$(taskset -pc $$)" "${extra[*]:-false}" > "$O/execution.txt"
if test "$SITE" = polaris; then
    "$V" -m pytest -q -p no:cacheprovider tests/test_reco_dispatch.py tests/test_e2e_bench.py \
        tests/test_coinc_attachment_jax.py tests/test_chain_reco_contract.py tests/test_prepared_context.py \
        tests/test_context_audit.py > "$O/gpu_tests.txt" 2>&1
fi
for compact in 0 1; do
    export CRVFASTSIM_COMPACT_DISPATCH=$compact
    tag=n50000_compact${compact}
    "$V" -m crvfastsim.analyses.e2e_bench "$L/gen.001400_00000011.root" \
        --library "$L/lib_prod_as_20files.root" --calib "$L/crvcalib_sim_best_v1_5_run1400.csv" \
        --deviation "$L/crvphoton_sim_best_v1_5_run1400.csv" --gdml "$L/mu2e_extracted_v04.gdml" \
        --run-config extracted_v04 --window offspill --seed 20261027 --jax "${extra[@]}" \
        --max-events 50000 --repeat "${AGENT13_REPEAT:-2}" --outdir "$L/results/$tag" > "$O/$tag.log" 2>&1
    cp "$L/results/$tag/benchmark.json" "$O/$tag.json"
done
"$V" scripts/agent13_compare_bench.py "$L/results/n50000_compact0" "$L/results/n50000_compact1" \
    --allow-derived-rounding --out "$O/n50000_regression.json"
