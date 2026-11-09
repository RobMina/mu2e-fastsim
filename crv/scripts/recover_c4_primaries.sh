#!/usr/bin/env bash
set -eo pipefail
: "${C4_CODE:?CRV checkout}"
: "${C4_CORE:?core checkout}"
: "${C4_FILES:?source ntuple path list}"
: "${C4_CACHED:?cached selection parquet}"
: "${C4_MCA:?original analysis checkout}"
: "${C4_OUT:?output directory}"
export PYTHONPATH="$C4_CODE:$C4_CORE"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m crvfastsim.veto.primaries --files "$C4_FILES" --cached "$C4_CACHED" \
  --mca "$C4_MCA" --detector-origin -3904 0 10171 --outdir "$C4_OUT"
