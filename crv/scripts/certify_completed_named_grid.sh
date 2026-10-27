#!/usr/bin/env bash
# Certify the already-submitted named-reference remainder without submission or regeneration.
set -eo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OLD=${1:?original continuation state}
OUT=${2:?new absolute certification output}
case "$OUT" in /exp/mu2e/data/users/ram2aq/crv_fastsim/*) ;; *) exit 2 ;; esac
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
muse setup ops
source /exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/site.env
set -euo pipefail
PY=/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python3
AREA=$M2FS_CODE/crv/art
mkdir "$OUT"
exec 9>"$CRVFS_DATA/thread_a_grid_continuation.lock"
flock -n 9 || { echo 'another controller owns the campaign'; exit 2; }
trap 'rc=$?; printf "exit=%s utc=%s\n" "$rc" "$(date -u +%FT%TZ)" > "$OUT/driver_exit.txt"' EXIT
cd "$OLD"
sha256sum -c code.sha256 > "$OUT/code_verification.txt"
sha256sum -c dumper.sha256 > "$OUT/dumper_verification.txt"
cp remainder_clusters.txt "$OUT/clusters.txt"
cd "$OUT"
printf 'phase=inventory utc=%s\n' "$(date -u +%FT%TZ)" | tee status.txt
timeout 180 jobsub_q -G mu2e --user ram2aq > queue.txt 2>&1
cat queue.txt
grep -q JOBSUBJOBID queue.txt
grep -q '0 total; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended' queue.txt
if awk '$1 ~ /^[0-9]+[.][0-9]+@/ {found=1} END {exit !found}' queue.txt; then
  echo 'jobs still queued: manual review required'; exit 2
fi
for cfg in run1a run2; do
  timeout 180 ifdh ls "$CRVFS_PNFS/g4ref_${cfg}_v1" > "inventory.$cfg.txt" 2>&1
  timeout 180 ifdh ls "$CRVFS_PNFS/g4ref_${cfg}_v1/logs" >> "inventory.$cfg.txt" 2>&1
  for k in $(seq 101 120); do
    seq=$(printf '001400_%08d' "$k")
    for kind in dts gen; do
      grep -Fxq "$CRVFS_PNFS/g4ref_${cfg}_v1/$kind.$CRVFS_OWNER.CosmicCRY${cfg}Ref.v1.$seq.art" "inventory.$cfg.txt"
    done
    grep -Fxq "$CRVFS_PNFS/g4ref_${cfg}_v1/logs/mu2e.$seq.log.gz" "inventory.$cfg.txt"
  done
  echo "$cfg all 20 art pairs and worker logs present"
done
printf 'phase=gridlogs utc=%s\n' "$(date -u +%FT%TZ)" | tee status.txt
while read -r cfg cluster schedd; do
  for k in $(seq 0 15); do
    timeout 180 jobsub_fetchlog -G mu2e --partial -J "$cluster.$k@$schedd" \
      --destdir "$OUT/gridlogs/$cfg/$k" > "$OUT/fetch.$cfg.$k.txt" 2>&1
    grep -R -F -x -q "=== grid_ref_sim: done shard $k" "$OUT/gridlogs/$cfg/$k"
    echo "completion marker: $cfg $cluster.$k@$schedd"
  done
done < clusters.txt
used=$(getfattr --only-values -n ceph.dir.rbytes "$M2FS_DATA")
quota=$(getfattr --only-values -n ceph.quota.max_bytes "$M2FS_DATA")
printf 'used=%s quota=%s\n' "$used" "$quota" > quota.txt
[ "$((quota-used))" -gt 2147483648 ] || { echo 'less than 2 GiB quota headroom'; exit 2; }
printf 'phase=dumping utc=%s\n' "$(date -u +%FT%TZ)" | tee status.txt
(
  set +u
  cd "$AREA"
  muse setup
  set -euo pipefail
  [ "$MUSE_WORK_DIR" = "$AREA" ] || exit 2
  bash "$CODE/scripts/dump_named_transport_reference.sh" "$OUT/tables" "$AREA" 105 120
) > dump.log 2>&1
printf 'phase=certifying utc=%s\n' "$(date -u +%FT%TZ)" | tee status.txt
export PYTHONPATH="$CODE:$M2FS_CODE/core${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m crvfastsim.analyses.reference_certification "$OUT/tables" \
  --first 105 --last 120 --expected-events 50000 --output "$OUT/certification.json" > certification.log 2>&1
cat certification.log
printf 'phase=COMPLETE remainder_generated_per_config=800000 utc=%s\n' "$(date -u +%FT%TZ)" | tee status.txt
