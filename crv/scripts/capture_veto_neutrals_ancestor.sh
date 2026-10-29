#!/bin/bash
# Capture one registered upstream recipe and its exact catalog observations, not physics.
set -euo pipefail
OUT=${1:?supply a new absolute documentary output directory}
DATA=/exp/mu2e/data/users/ram2aq/crv_veto_r3_coverage_20261028
case "$OUT" in "$DATA"/*) ;; *) echo 'Output must be under the dedicated data area' >&2; exit 2 ;; esac
ARCHIVE=/pnfs/mu2e/persistent/datasets/phy-etc/cnf/mu2e/NeutralsCat/MDC2025ab/tar/f9/b7/cnf.mu2e.NeutralsCat.MDC2025ab.0.tar
PIN=22dc4c2e314dfb9da041c25856cb361dffa02c38f969ddf901c6b56f67749bd3
mkdir "$OUT"
cp "$0" "$OUT/capture.sh"
printf '%s  %s\n' "$PIN" "$ARCHIVE" > "$OUT/archive_sha256.txt"
sha256sum -c "$OUT/archive_sha256.txt"
tar -tvf "$ARCHIVE" > "$OUT/members.txt"
tar -xOf "$ARCHIVE" jobpars.json | head -c 2097153 > "$OUT/jobpars.json"
test "$(wc -c < "$OUT/jobpars.json")" -eq 1340500
tar -xOf "$ARCHIVE" mu2e.fcl | head -c 8193 > "$OUT/mu2e.fcl"
test "$(wc -c < "$OUT/mu2e.fcl")" -eq 197
for name in runtime32644_metadata_v1 neutralscat_ab_jobdef_metadata_v1 neutralscat_ab_jobdef_location_v1; do
  test "$(wc -c < "$DATA/$name.log")" -le 1048576
  sha256sum "$DATA/$name.log" >> "$OUT/catalog_original_sha256.txt"
  cp "$DATA/$name.log" "$OUT/$name.log"
done
jq '{code,setup,jobname,tbs:(.tbs | del(.inputs)),inputs:(.tbs.inputs | to_entries | map({key:.key,group_size:.value[0],filename_count:(.value[1]|length),first:.value[1][0],last:.value[1][-1]}))}' "$OUT/jobpars.json" > "$OUT/recipe_summary.json"
sha256sum -c "$OUT/archive_sha256.txt"
sha256sum -c "$OUT/catalog_original_sha256.txt"
cmp "$0" "$OUT/capture.sh"
(cd "$OUT" && sha256sum capture.sh archive_sha256.txt members.txt jobpars.json mu2e.fcl catalog_original_sha256.txt runtime32644_metadata_v1.log neutralscat_ab_jobdef_metadata_v1.log neutralscat_ab_jobdef_location_v1.log recipe_summary.json > files_sha256.txt && sha256sum -c files_sha256.txt)
printf '%s\n' 'Documentary capture only: no worker execution, physical ancestry certification or exposure weight.'
