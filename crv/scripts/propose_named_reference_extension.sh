#!/bin/bash
# Print the bounded grid continuation proposal; this script never stages, submits or cancels anything.
set -euo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
P=${CRVFS_PNFS:?source site.env}
OWNER=${CRVFS_OWNER:?source site.env}
MODE=${1:-pilot}
case "$MODE" in
  pilot) FIRST=102; NJOBS=3 ;;
  remainder) FIRST=105; NJOBS=16 ;;
  *) echo 'usage: propose_named_reference_extension.sh pilot|remainder'; exit 2 ;;
esac
printf '# PROPOSAL ONLY: %s, %s jobs/config, 50000 generated/job, no tarball/build.\n' "$MODE" "$NJOBS"
printf '# Prerequisite: user approval, fresh queue/output dedup, staged FCL hashes unchanged.\n'
printf '# Remainder additionally requires the 3-job/config pilot logs and persisted count certification.\n'
printf '# The successful 50k pilots measured 54:34 (Run1A) and 68:52 (Run2); no runtime extrapolation.\n'
for cfg in run1a run2; do
  if [ "$cfg" = run1a ]; then BASE=20264100; else BASE=20265100; fi
  OUT=$P/g4ref_${cfg}_v1
  SCOPE=${OUT/#\/pnfs\/mu2e/\/mu2e}
  printf '\n# %s: subruns %s..%s, seeds %s..%s\n' "$cfg" "$FIRST" "$((FIRST+NJOBS-1))" "$((BASE+FIRST-1))" "$((BASE+FIRST+NJOBS-2))"
  printf 'jobsub_submit -N %s -G mu2e --role=Analysis --expected-lifetime=4h \\\n' "$NJOBS"
  printf '  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC \\\n'
  printf '  --lines=%q --memory=3000MB --disk=4000MB \\\n' '+FERMIHTC_AutoRelease=True'
  printf '  --need-storage-modify=%q \\\n' "$SCOPE"
  printf '  -e NEVTS=50000 -e BASESEED=%s -e FIRST_SUBRUN=%s -e TAG=v1 -e CFG=%s \\\n' "$BASE" "$FIRST" "$cfg"
  printf '  -e FCL_PNFS=%q -e MUSING=MDC2025ay -e OWNER=%q \\\n' "$OUT/fcl" "$OWNER"
  printf '  %q %q\n' "file://$CODE/grid/grid_ref_sim.sh" "$OUT"
done
