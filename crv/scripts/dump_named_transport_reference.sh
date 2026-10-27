#!/bin/bash
# Dump existing named-reference art pairs directly from pnfs; no build or event generation.
set -eo pipefail
OUT=${1:?new absolute output directory}
AREA=${2:?existing built CrvFastSim art area}
FIRST=${3:-101}
LAST=${4:-101}
case "$OUT" in /*) ;; *) echo 'output must be absolute'; exit 2 ;; esac
[ "${MUSE_WORK_DIR:-}" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR=${MUSE_WORK_DIR:-}"; exit 2; }
PNFS=${CRVFS_PNFS:?source site.env}
OWNER=${CRVFS_OWNER:?source site.env}
mkdir "$OUT"
printf 'area=%s\nMUSE_WORK_DIR=%s\nfirst=%s\nlast=%s\n' "$AREA" "$MUSE_WORK_DIR" "$FIRST" "$LAST" > "$OUT/dump_manifest.txt"
sha256sum "$AREA/CrvFastSim/src/CrvStepTableDumper_module.cc" \
  "$AREA/CrvFastSim/fcl/dump_crvsteps.fcl" "$AREA/CrvFastSim/fcl/dump_gen.fcl" \
  "$AREA"/build/*/CrvFastSim/lib/libcrvfastsim_CrvStepTableDumper_module.so >> "$OUT/dump_manifest.txt"
for cfg in run1a run2; do
  mkdir "$OUT/$cfg"
  for k in $(seq "$FIRST" "$LAST"); do
    S=$(printf '001400_%08d' "$k")
    Q=$OUT/$cfg
    IN=$PNFS/g4ref_${cfg}_v1
    gzip -dc "$IN/logs/mu2e.$S.log.gz" > "$Q/worker.$S.log"
    grep -q 'Exit status: 0' "$Q/worker.$S.log" || { echo 'worker did not exit zero'; exit 2; }
    for kind in dts gen; do
      if [ "$kind" = dts ]; then STEM=crvsteps; FCL=dump_crvsteps; else STEM=gen; FCL=dump_gen; fi
      INPUT=$IN/$kind.$OWNER.CosmicCRY${cfg}Ref.v1.$S.art
      printf 'input=%s\n' "$INPUT" >> "$OUT/dump_manifest.txt"
      rc=0
      mu2e -c CrvFastSim/fcl/$FCL.fcl -s "$INPUT" -T "$Q/$STEM.$S.root" > "$Q/$FCL.$S.log" 2>&1 || rc=$?
      printf 'dump_exit=%s\n' "$rc" >> "$Q/$FCL.$S.log"
      [ "$rc" = 0 ] || { tail -40 "$Q/$FCL.$S.log"; exit "$rc"; }
      grep -E 'TrigReport Events total|dump_exit' "$Q/$FCL.$S.log"
    done
    sha256sum "$Q/worker.$S.log" "$Q/crvsteps.$S.root" "$Q/gen.$S.root" >> "$OUT/dump_manifest.txt"
  done
done
