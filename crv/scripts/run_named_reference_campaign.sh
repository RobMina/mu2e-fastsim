#!/usr/bin/env bash
# Run frozen named configurations serially after the completed reference certificates exist.
set -eo pipefail
CFG=${1:?run1a, run2 or all}
OUT=${2:?new absolute output directory}
case "$CFG" in run1a|run2|all) ;; *) exit 2 ;; esac
case "$OUT" in /exp/mu2e/data/users/ram2aq/crv_fastsim/*) ;; *) exit 2 ;; esac
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
source /exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/site.env
PY=/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python3
set -euo pipefail
CODE=/exp/mu2e/app/users/ram2aq/analysis/crv-thread-a-validation
CORE=/exp/mu2e/app/users/ram2aq/analysis/core-thread-a-validation
D=$CRVFS_DATA
export PYTHONPATH="$CORE:$CODE"
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[ ! -e "$OUT" ] || { echo 'output already exists'; exit 2; }
[ "$(git -C "$CORE" rev-parse HEAD)" = 2bcefdee2a521fa5531f8072c108dd8176ad8610 ]
[ "$(git -C "$CODE" rev-parse HEAD)" = 3895aae8cf0cb09f3ae1e00a77a90e7ad650dc2d ]
cd "$D"
sha256sum -c <<'HASHES'
556524c8a1a8bf272d898b9963f2af1092664e59a3906a22d334de4ace0b95ae  thread_a_paired70k_v3_20261025/run1a/audit50k.json
0cdd0f492304998950f0e06c1d1519535dba4220827fa8b61f875b14042f89b9  thread_a_paired70k_v3_20261025/run2/audit50k.json
6bec5436323426b32737c4af5cebfa25fd86c272da1b554617bd9f6e3a585d47  thread_a_paths70k_final_20261025/run1a/paths50k.json
08c8460eee8e7d9af7cb38e64e61026651ff7c9adedbac366e43178cd72c8880  thread_a_paths70k_final_20261025/run2/paths50k.json
HASHES
FINISH=$D/thread_a_grid_completion_20261026
DEADLINE=$((SECONDS+7200))
while [ ! -f "$FINISH/driver_exit.txt" ]; do
  [ "$SECONDS" -lt "$DEADLINE" ] || { echo 'certification wait deadline exceeded'; exit 2; }
  echo "waiting for reference certification $(date -u +%FT%TZ)"
  sleep 60
done
grep -q '^exit=0 ' "$FINISH/driver_exit.txt"
grep -q '^phase=COMPLETE ' "$FINISH/status.txt"
[ -s "$FINISH/certification.json" ]
used=$(getfattr --only-values -n ceph.dir.rbytes "$M2FS_DATA")
quota=$(getfattr --only-values -n ceph.quota.max_bytes "$M2FS_DATA")
printf 'used=%s quota=%s\n' "$used" "$quota"
[ "$((quota-used))" -gt 3221225472 ] || { echo 'less than 3 GiB quota headroom'; exit 2; }
sha256sum "$0" "$CODE/crvfastsim/analyses/named_reference_campaign.py"
"$PY" -m crvfastsim.analyses.named_reference_campaign \
  --certificates "$D/thread_a_certification_baseline_20261025.json" \
    "$D/thread_a_grid_continuation_20261025/pilot/certification.json" "$FINISH/certification.json" \
  --retained-candidates "$D/thread_a_named70k_20261025" \
  --retained-audits "$D/thread_a_paired70k_v3_20261025" \
  --retained-paths "$D/thread_a_paths70k_final_20261025" \
  --geometry-dir "$D/geometry" --config "$CFG" --outdir "$OUT"
