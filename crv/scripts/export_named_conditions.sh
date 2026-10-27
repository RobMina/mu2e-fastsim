#!/usr/bin/env bash
# Export and inspect the two explicit stock text-overlay bundles; requires the Python analysis environment.
set -eo pipefail
SOURCE=${1:?stock dump root}
OUT=${2:?new output root}
CODE=${3:?CRV code tree}
CORE=${4:?explicit core tree}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CORE:$CODE"
mkdir "$OUT"
for cfg in run1a run2; do
  rc=run1a_v01; [[ "$cfg" = run2 ]] && rc=run2_v10
  python3 -m crvfastsim.named_conditions export --source "$SOURCE/$cfg" --run-config "$rc" --outdir "$OUT/$cfg"
  python3 -m crvfastsim.analyses.named_inventory --manifest "$OUT/$cfg/manifest.json" \
    --reference "$SOURCE/$cfg/crvreco.root" --out "$OUT/${cfg}_inventory.json"
done
sha256sum "$OUT"/*/manifest.json "$OUT"/*/calib.csv "$OUT"/*/photon.csv > "$OUT/sha256.txt"
