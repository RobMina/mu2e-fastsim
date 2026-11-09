#!/usr/bin/env bash
# Run stock CRV response and its existing dumper with an explicit named text-condition overlay.
set -eo pipefail
OUT=${1:?new absolute output directory}
AREA=${2:?existing built CrvFastSim area}
CODE=${3:?source tree with named FCL}
NEVENTS=${4:-1}
[[ "$OUT" = /* ]] || exit 2
[[ "${MUSE_WORK_DIR:-}" = "$AREA" ]] || { echo "wrong MUSE_WORK_DIR"; exit 2; }
mkdir "$OUT"
export FHICL_FILE_PATH="$CODE/art/CrvFastSim/fcl:$CODE/grid/fcl:$FHICL_FILE_PATH"
export MU2E_SEARCH_PATH="$CODE/grid/fcl:$MU2E_SEARCH_PATH"
OFFLINE=$(readlink -f "$AREA/backing/backing/Offline")
for cfg in run1a run2; do
  Q="$OUT/$cfg"
  mkdir "$Q"
  IN="${CRVFS_PNFS:?source site.env}/g4ref_${cfg}_v1/dts.${CRVFS_OWNER:?source site.env}.CosmicCRY${cfg}Ref.v1.001400_00000101.art"
  FCL="$CODE/art/CrvFastSim/fcl/dump_named_crvreco_$cfg.fcl"
  printf 'input=%s\narea=%s\noffline=%s\n' "$IN" "$AREA" "$OFFLINE" > "$Q/manifest.txt"
  sha256sum "$FCL" "$CODE/art/CrvFastSim/fcl/dump_named_crvreco.fcl" "$AREA/CrvFastSim/src/CrvRecoTableDumper_module.cc" "$AREA"/build/*/CrvFastSim/lib/libcrvfastsim_CrvRecoTableDumper_module.so >> "$Q/manifest.txt"
  geom=run1a_v01; [[ "$cfg" = run2 ]] && geom=nominal_v10
  for prefix in calib status photonyield; do
    f="$OFFLINE/CRVConditions/data/${prefix}_${geom}.txt"
    cp "$f" "$Q/${prefix}_source.txt"
    sha256sum "$f" >> "$Q/manifest.txt"
  done
  cp "$OFFLINE/CRVConditions/data/$geom.txt" "$Q/ordinal_source.txt"
  sha256sum "$OFFLINE/CRVConditions/data/$geom.txt" >> "$Q/manifest.txt"
  counters=crv_counters_run1a_v01.txt; [[ "$cfg" = run2 ]] && counters=crv_counters_v10.txt
  sha256sum "$OFFLINE/Mu2eG4/geom/$counters" >> "$Q/manifest.txt"
  mu2e --debug-config "$Q/resolved.fcl" -c "$FCL" --annotate > "$Q/expand.log" 2>&1
  /usr/bin/time -v mu2e -c "$FCL" -s "$IN" -n "$NEVENTS" -T "$Q/crvreco.root" > "$Q/stock.log" 2>&1 || { tail -60 "$Q/stock.log"; exit 1; }
  sha256sum "$Q/resolved.fcl" "$Q/crvreco.root" "$Q/stock.log" >> "$Q/manifest.txt"
  grep -E 'TrigReport Events total|CRV|reading from|Elapsed|Exit status' "$Q/stock.log" | tail -35
done
