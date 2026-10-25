#!/bin/bash
# Compare original shard histograms, never the legacy flow-dropping merges; run in an ana session.
# Usage: audit_pilot.sh PILOT_OUT A0G_REF [FROZEN_A1 FROZEN_E2E]
set -e
OUT=${1:?pilot output}; REF=${2:?A0g reference}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$CODE:${M2FS_CORE:?isolated core checkout}" PYTHONDONTWRITEBYTECODE=1
AL=(--alias CrvSteps_fastCrvSteps_noName=CrvSteps_steps_noName --alias CrvSteps_compressDetStepMCs_noName=CrvSteps_steps_noName)
COUNTS=(--count-key crvRecoDQM/NPc --count-key crvRecoDQM/PEc --count-key crvRecoDQM/X_extracted \
        --count-key crvRecoDQM/Y_extracted --count-key crvRecoDQM/Z_extracted \
        --count-key dqmCrv/NPc --count-key dqmCrv/PEc --count-key dqmCrv/X --count-key dqmCrv/Y --count-key dqmCrv/Z)
for arm in a1 e2e; do
  python3 -m valhist.coverage "$REF" "$OUT/$arm/valhist.root" --outdir "$OUT/coverage_${arm}_v2" "${AL[@]}" "${COUNTS[@]}"
done
python3 -m valhist.coverage "$OUT/a1/valhist.root" "$OUT/e2e/valhist.root" --outdir "$OUT/coverage_e2e_vs_a1" "${AL[@]}" "${COUNTS[@]}"
python3 -m valhist.paired_rates "$OUT/a1/valhist.root" "$OUT/e2e/valhist.root" --out "$OUT/paired_e2e_vs_a1.json"
if [ "$#" -ge 4 ]; then
  python3 -m valhist.coverage "$3" "$OUT/a1/valhist.root" --outdir "$OUT/coverage_a1_vs_frozen" "${AL[@]}" "${COUNTS[@]}"
  python3 -m valhist.coverage "$4" "$OUT/e2e/valhist.root" --outdir "$OUT/coverage_e2e_vs_frozen" "${AL[@]}" "${COUNTS[@]}"
fi
