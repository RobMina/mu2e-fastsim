#!/bin/bash
# One sequential analysis job: exact sampler moments, donor-file variation and corrected source preparation.
set -eo pipefail
OUT=${1:?new study output directory}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CORE=${M2FS_CORE:?core checkout}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
pyenv ana
export PYTHONPATH="$CODE:$CORE" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"
mkdir "$OUT"
trap 'result=$?; printf "%s\n" "$result" > "$OUT/exit"' EXIT
ROOT=/exp/mu2e/data/users/ram2aq/crv_genassoc_20261027
DATA=/exp/mu2e/data/users/ram2aq/crv_fastsim
python3 -m valhist.stepclosure shower-audit "$ROOT/source_train_strict_v2/library/library.json" \
  --out "$OUT/source_association_audit.json"
python3 -m valhist.shower_uncertainty prepare "$ROOT/strict_v2" "$OUT/production"
python3 -m valhist.shower_uncertainty prepare "$ROOT/source_train_strict_v2" "$OUT/reference_source"
python3 -m valhist.shower_uncertainty evaluation "$OUT/production" "$OUT/evaluation" \
  --corrected "$ROOT/strict_v2/validation/s11-20/ccp/fast1b.{s}.root" \
  --reference "$DATA/valhist/g4ref_v1/g4all_tables/g4all.{s}.root" \
  --generated "$DATA/g4ref_v1_tables/gen.{s}.root" --shards 11-20
python3 -m valhist.shower_uncertainty_report "$OUT/production" "$OUT/evaluation" "$OUT/production_result"
python3 -m valhist.shower_uncertainty sampler-check "$OUT/production" \
  "$DATA/g4ref_v1_tables/gen.001400_00000011.root" --out "$OUT/sampler_check.json" --draws 256
