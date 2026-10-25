#!/bin/bash
# Fixed-code extracted candidate campaign; reference tables are read-only and the output tag must be new.
set -eo pipefail
D=${CRVFS_DATA:?set CRVFS_DATA}
OUT=${1:?new absolute output directory}
FIRST=${2:-1}
LAST=${3:-20}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir "$OUT"
mkdir "$OUT/tables" "$OUT/audit" "$OUT/v1_batches"
printf 'candidate_crv=%s\nPYTHONPATH=%s\nfirst=%s\nlast=%s\n' "$(git -C "$CODE" rev-parse HEAD)" "$PYTHONPATH" "$FIRST" "$LAST" > "$OUT/manifest.txt"
sha256sum "$CODE/crvfastsim/transport.py" "$CODE/crvfastsim/analyses/transport_validation.py" "$D/geometry/mu2e_extracted_v04.gdml" >> "$OUT/manifest.txt"
for k in $(seq "$FIRST" "$LAST"); do
  S=$(printf '001400_%08d' "$k")
  GEN=$D/g4ref_v1_tables/gen.$S.root
  REF=$D/g4ref_v1_tables/crvsteps.$S.root
  BASE=$D/fixvar/v1_absorbsk/tables/fastC.$S.root
  FAST=$OUT/tables/fastC.$S.root
  /usr/bin/time -f 'transport_wall_s=%e maxrss_kb=%M' python3 -m crvfastsim transport "$GEN" --stage C --seed "$k" \
    --run-config extracted_v04 --gdml "$D/geometry/mu2e_extracted_v04.gdml" --outdir "$OUT/tables" > "$OUT/transport_$k.log" 2>&1
  python3 -m crvfastsim.analyses.transport_validation --gen "$GEN" --reference "$REF" --baseline "$BASE" --candidate "$FAST" \
    --config extracted_v04 --expected-events 50000 --gdml "$D/geometry/mu2e_extracted_v04.gdml" \
    --output "$OUT/audit/shard_$k.json" > "$OUT/audit_$k.log" 2>&1
  python3 -m crvfastsim v1compare "$FAST" --compare "$REF" --run-config extracted_v04 \
    --emit "$OUT/v1_batches/shard_$k.json" > "$OUT/v1_$k.log" 2>&1
  echo "completed shard $k $(date -Iseconds)"
done
python3 -m crvfastsim v1compare "$OUT/v1_batches/*.json" --merge --run-config extracted_v04 --outdir "$OUT/v1" > "$OUT/v1.log" 2>&1
cat "$OUT/v1.log"
echo "campaign complete $(date -Iseconds)"
