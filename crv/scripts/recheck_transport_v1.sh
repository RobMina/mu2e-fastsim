#!/bin/bash
# Recheck persisted Thread A tables with current V1 gates without rerunning transport.
set -eo pipefail
D=${CRVFS_DATA:?set CRVFS_DATA}
OLD=${1:?existing Thread A campaign}
OUT=${2:?new absolute output directory}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir "$OUT"
printf 'crv_head=%s\nPYTHONPATH=%s\nsource_campaign=%s\n' "$(git -C "$CODE" rev-parse HEAD)" "$PYTHONPATH" "$OLD" > "$OUT/manifest.txt"
sha256sum "$CODE/crvfastsim/analyses/v1compare.py" "$CODE/crvfastsim/cli.py" >> "$OUT/manifest.txt"
for cfg in run1a run2 extracted; do
  if [ "$cfg" = run1a ]; then
    RC=run1a_v01; S=001400_00009111
  elif [ "$cfg" = run2 ]; then
    RC=run2_v10; S=001400_00009112
  else
    RC=extracted_v04
  fi
  rc=0
  if [ "$cfg" = extracted ]; then
    python3 -m crvfastsim v1compare "$OLD/extracted20/v1_batches/*.json" --merge --run-config "$RC" \
      --outdir "$OUT/$cfg" > "$OUT/$cfg.log" 2>&1 || rc=$?
  else
    python3 -m crvfastsim v1compare "$OLD/${cfg}_tables/fastC.$S.root" \
      --compare "$D/g4ref_test_$cfg/crvsteps.$S.root" --run-config "$RC" \
      --outdir "$OUT/$cfg" > "$OUT/$cfg.log" 2>&1 || rc=$?
  fi
  printf '%s exit=%s\n' "$cfg" "$rc" | tee -a "$OUT/exit_codes.txt"
  cat "$OUT/$cfg.log"
  if [ "$rc" -gt 1 ] || [ ! -f "$OUT/$cfg/v1compare_stats.json" ]; then exit 2; fi
  sha256sum "$OUT/$cfg/v1compare_stats.json" >> "$OUT/manifest.txt"
done
