#!/usr/bin/env bash
# Read resolved simulation conditions and versioned stock assets without modifying the database.
set -eo pipefail
OUT=${1:?new absolute output directory}
AREA=${2:?existing built CrvFastSim area}
[[ "$OUT" = /* ]] || exit 2
[[ "${MUSE_WORK_DIR:-}" = "$AREA" ]] || { echo "wrong MUSE_WORK_DIR"; exit 2; }
mkdir "$OUT"
OFFLINE=$(readlink -f "$AREA/backing/backing/Offline")
printf 'area=%s\noffline=%s\n' "$AREA" "$OFFLINE" > "$OUT/manifest.txt"
for run in 1400 1430 1440; do
  for table in CRVSiPM CRVTime CRVBadChan CRVPhoton; do
    dbTool print-run --purpose Sim_best --version v1_5 --run "$run:101" --table "$table" > "$OUT/${table}_run${run}.txt" 2>&1
  done
done
for geometry in run1a_v01 nominal_v10; do
  for prefix in calib status photonyield; do
    f="$OFFLINE/CRVConditions/data/${prefix}_${geometry}.txt"
    sha256sum "$f" >> "$OUT/manifest.txt"
    head -5 "$f"
  done
  sha256sum "$OFFLINE/CRVConditions/data/$geometry.txt" >> "$OUT/manifest.txt"
done
cat "$OUT"/CRV*_run*.txt
sha256sum "$AREA/CrvFastSim/src/CrvRecoTableDumper_module.cc" "$AREA"/build/*/CrvFastSim/lib/libcrvfastsim_CrvRecoTableDumper_module.so >> "$OUT/manifest.txt"
