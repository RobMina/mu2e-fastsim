#!/bin/bash
# Validate corrected-library steps against retained G4 and CCP baselines; no transport or build is run.
set -eo pipefail
ROOT=$(cd "${1:?campaign directory}" && pwd); SHARDS=${2:-11-20}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CODE"
python3 -m valhist.genassoc_campaign verify-library "$ROOT"
D=${CRVFS_DATA:?set CRVFS_DATA to the CRV data root (see site.env.example)}
OLD=${STEP_LAYERS_OUT:?set STEP_LAYERS_OUT to the step-layer study directory}
F=$D/thread_a_ledger_20261025_c3895aae/extracted20/tables
G=$D/valhist/g4ref_v1/g4all_tables
L=$ROOT/library/library.root
OUT=$ROOT/validation/s${SHARDS}
test -s "$L" || { echo "corrected library missing" >&2; exit 2; }
mkdir "$OUT"
FASTC_DIR="$F" G4_DIR="$G" bash scripts/run_step_layers.sh "$L" "$OUT/ccp" "$SHARDS" \
  --crossing-context --contain-steps --point-placement
python3 -m valhist.stepclosure exposure-guard --generated "$D/g4ref_v1_tables/gen.{s}.root" \
  --arm "reference=$G/g4all.{s}.root" --arm "corrected=$OUT/ccp/fast1b.{s}.root" \
  --arm "old=$OLD/fast1b_ccp/fast1b.{s}.root" --shards "$SHARDS" --out "$OUT/exposure.json"
python3 -m valhist.stepclosure primary-guard --base "$OLD/fast1b_ccp/fast1b.{s}.root" \
  --candidate "$OUT/ccp/fast1b.{s}.root" --shards "$SHARDS" --descendants --out "$OUT/muon_family_guard.json"
python3 -m valhist.steplayers posyz --arm "a0g=$G/g4all.{s}.root" \
  --arm "old=$OLD/fast1b_ccp/fast1b.{s}.root" --arm "corrected=$OUT/ccp/fast1b.{s}.root" \
  --shards "$SHARDS" --out "$OUT/position_comparison.json"
python3 -m valhist.steplayers report "$OUT/ccp/breakdown.json" --views grp,grp_lay_cls,grp_cls_outside \
  > "$OUT/breakdown.txt"
lo=${SHARDS%-*}; hi=${SHARDS#*-}; candidate=(); reference=()
for k in $(seq "$lo" "$hi"); do
  s=$(printf '001400_%08d' "$k")
  candidate+=("$OUT/ccp/fast1b.$s.root"); reference+=("$G/g4all.$s.root")
done
python3 -m crvfastsim v1compare "${candidate[@]}" --compare "${reference[@]}" --outdir "$OUT/v1"
python3 -m crvfastsim tier1b-check "${candidate[@]}" --compare "${reference[@]}" --outdir "$OUT/v3"
python3 -m valhist.steplayers response --ref "$G/g4all.{s}.root" --cand "$OUT/ccp/fast1b.{s}.root" \
  --shards "$SHARDS" --calib "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv" \
  --deviation "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" --out "$OUT/response.json"
python3 -m valhist.steplayers clusters "$OUT/response.json" > "$OUT/clusters.txt"
echo "completed $OUT"
