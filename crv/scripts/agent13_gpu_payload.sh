#!/bin/bash
set -eo pipefail
L=${1:?}; O=${2:?}; SIF=${3:?}
A() {
    apptainer exec --nv --cleanenv \
      --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},PYTHONPATH=$L/code/crv:$L/code/core,CRVFS_LOOKUP_DIR=$L,CRVFS_DATA=$L,JAX_ENABLE_X64=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,PYTHONDONTWRITEBYTECODE=1,CRVFASTSIM_COMPACT_DISPATCH=${COMPACT:-0}" \
      -B "$L" -B "$O" --pwd "$L/code/crv" "$SIF" "$@"
}
A python3 -m pytest -q -p no:cacheprovider tests/test_reco_dispatch.py tests/test_e2e_bench.py \
    tests/test_coinc_attachment_jax.py tests/test_chain_reco_contract.py tests/test_prepared_context.py \
    tests/test_context_audit.py tests/test_agent13_context_campaign.py > "$O/gpu_tests.txt" 2>&1
if test "${AGENT13_CONTEXT_CAMPAIGN:-0}" = 1; then
    A python3 scripts/agent13_context_campaign.py --data "$L" --inputs-dir "$L" \
        --library "$L/lib_prod_as_20files.root" --calib "$L/crvcalib_sim_best_v1_5_run1400.csv" \
        --deviation "$L/crvphoton_sim_best_v1_5_run1400.csv" --gdml "$L/mu2e_extracted_v04.gdml" \
        --outdir "$L/results" --summary-dir "$O"
    exit 0
fi
for n in 50000; do
    for compact in 0 1; do
        export COMPACT=$compact
        tag=n${n}_compact${compact}
        A python3 -m crvfastsim.analyses.e2e_bench "$L/gen.001400_00000011.root" \
            --library "$L/lib_prod_as_20files.root" --calib "$L/crvcalib_sim_best_v1_5_run1400.csv" \
            --deviation "$L/crvphoton_sim_best_v1_5_run1400.csv" --gdml "$L/mu2e_extracted_v04.gdml" \
            --run-config extracted_v04 --window offspill --seed 20261027 --jax \
            --max-events "$n" --repeat 2 --outdir "$L/results/$tag" > "$O/$tag.log" 2>&1
        cp "$L/results/$tag/benchmark.json" "$O/$tag.json"
    done
    A python3 scripts/agent13_compare_bench.py "$L/results/n${n}_compact0" "$L/results/n${n}_compact1" \
        --allow-derived-rounding --out "$O/n${n}_regression.json"
done
