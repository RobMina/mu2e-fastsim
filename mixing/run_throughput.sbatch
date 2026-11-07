#!/bin/bash
# Single-pass end-to-end throughput: three fresh fast_detectors processes (a fresh private JAX/CUDA cache, then two
# processes sharing the populated caches), each timed launch-to-exit, followed by the mixing.throughput report.
# Account, partition and GPU are chosen at submit time:
#   sbatch -A ACCOUNT -p PARTITION --gres=gpu:1 mixing/run_throughput.sbatch OUT REFERENCE STOCK -- INPUTS...
# OUT: new output directory. REFERENCE: saved direct reference run. STOCK: saved stock run (art.time, art.log,
# response.root). INPUTS: direct-frame fast_detectors inputs (--frames-manifest, --expanded, --truth, --stock,
# --photon-csv, --tracker-conditions, ...); the single-pass flags and --out are added here.
# Environment: FASTSIMKIT_SIF (image), M2FS_CODE (umbrella checkout), THROUGHPUT_BINDS (comma-separated host
# directories holding the code, inputs and OUT), MU2E_SEARCH_PATH, CRVFS_LOOKUP_DIR. Optional: THROUGHPUT_PYTHONPATH,
# THROUGHPUT_HARDWARE_LABEL (public fast label; default from the allocated GPU, CPUs and memory),
# THROUGHPUT_STOCK_LABEL (public stock label; without it the stock hardware is "not recorded").
#SBATCH --job-name=fast_throughput
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -eo pipefail
[[ $# -ge 4 && $4 == -- ]] || { echo 'usage: run_throughput.sbatch OUT REFERENCE STOCK -- INPUTS...' >&2; exit 2; }
: "${FASTSIMKIT_SIF:?set the Apptainer image}"
: "${M2FS_CODE:?set the umbrella checkout}"
: "${THROUGHPUT_BINDS:?set the host directories to bind}"
: "${MU2E_SEARCH_PATH:?set the Offline data search path}"
: "${CRVFS_LOOKUP_DIR:?set the CRV lookup directory}"
command -v apptainer > /dev/null || module load apptainer
REFERENCE=$(readlink -f "$2")
STOCK=$(readlink -f "$3")
mkdir "$1"
OUT=$(cd "$1" && pwd -P)
shift 4
mkdir "$OUT/jax_cache" "$OUT/cuda_cache"

common=(--cleanenv -B "$THROUGHPUT_BINDS"
  --env "PYTHONPATH=${THROUGHPUT_PYTHONPATH:-$M2FS_CODE:$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker}"
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env PYTHONDONTWRITEBYTECODE=1 --env CRVFS_NO_FETCH=1
  --env "MU2E_SEARCH_PATH=$MU2E_SEARCH_PATH" --env "CRVFS_LOOKUP_DIR=$CRVFS_LOOKUP_DIR"
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false
  --env "JAX_COMPILATION_CACHE_DIR=$OUT/jax_cache" --env "CUDA_CACHE_PATH=$OUT/cuda_cache")
label_json() { apptainer exec --cleanenv "$FASTSIMKIT_SIF" python3 -c \
  'import json, sys; print(json.dumps({"label": sys.argv[1]}))' "$1"; }

# Raw provenance stays private in hardware.txt; only the one-line labels reach the report's public numbers.
{ date -u +%FT%TZ; lscpu; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; } > "$OUT/hardware.txt"
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
label="$gpu; ${SLURM_CPUS_PER_TASK:-unknown} CPUs${SLURM_MEM_PER_NODE:+; $((SLURM_MEM_PER_NODE / 1024)) GB RAM}"
label_json "${THROUGHPUT_HARDWARE_LABEL:-$label}" > "$OUT/hardware.json"
[[ -z ${THROUGHPUT_STOCK_LABEL:-} ]] || label_json "$THROUGHPUT_STOCK_LABEL" > "$OUT/stock_hardware.json"

format='{"wall_s": %e, "user_s": %U, "system_s": %S, "max_rss_kb": %M, "exit_status": %x}'
for name in fresh_cache populated_cache_1 populated_cache_2; do
  find "$OUT/jax_cache" "$OUT/cuda_cache" -type f -printf '%p %s\n' > "$OUT/$name.cache_before.txt"
  /usr/bin/time -f "$format" -o "$OUT/$name.time.json" \
    apptainer exec --nv "${common[@]}" "$FASTSIMKIT_SIF" \
    python3 -m mixing.process_timer --receipt "$OUT/$name.process.time.json" -- \
    python3 -m mixing.fast_detectors "$@" --repeats 1 --preparation-repeats 1 --skip-comparisons \
    --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  find "$OUT/jax_cache" "$OUT/cuda_cache" -type f -printf '%p %s\n' > "$OUT/$name.cache_after.txt"
done
apptainer exec "${common[@]}" "$FASTSIMKIT_SIF" python3 -m mixing.throughput "$OUT" \
  --reference "$REFERENCE" --stock "$STOCK" --out "$OUT/throughput.json"
