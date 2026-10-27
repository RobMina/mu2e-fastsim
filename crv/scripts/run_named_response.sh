#!/usr/bin/env bash
# A3 and A4 diagnostic arms on exactly the stock event universe, including events without steps.
set -eo pipefail
STOCK=${1:?stock reference directory}
CONDITIONS=${2:?conditions bundle root}
OUT=${3:?new output root}
CODE=${4:?CRV source tree}
CORE=${5:?explicit core tree}
STEPS=${6:?named reference step root}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CORE:$CODE"
mkdir "$OUT"
for cfg in run1a run2; do
  rc=run1a_v01; [[ "$cfg" = run2 ]] && rc=run2_v10
  mkdir "$OUT/$cfg"
  REF="$STOCK/$cfg/crvreco.root"
  MAN="$CONDITIONS/$cfg/manifest.json"
  python3 -m valhist.fasttables reco "$REF" "$OUT/$cfg/a3.root" --run-config "$rc" --conditions-manifest "$MAN" > "$OUT/$cfg/a3.log" 2>&1
  python3 -m crvfastsim v2compare "$OUT/$cfg/a3.root" --compare "$REF" --run-config "$rc" --outdir "$OUT/$cfg/a3_v2" > "$OUT/$cfg/a3_v2.log" 2>&1
  python3 -m valhist.fasttables chain "$STEPS/$cfg/crvsteps.001400_00000101.root" "$OUT/$cfg/a4.root" \
    --event-reference "$REF" --run-config "$rc" --conditions-manifest "$MAN" > "$OUT/$cfg/a4.log" 2>&1
  python3 -m crvfastsim v2compare "$OUT/$cfg/a4.root" --compare "$REF" --run-config "$rc" --outdir "$OUT/$cfg/a4_v2" > "$OUT/$cfg/a4_v2.log" 2>&1
  cat "$OUT/$cfg/a3_v2.log" "$OUT/$cfg/a4_v2.log"
done
