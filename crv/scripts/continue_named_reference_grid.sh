#!/usr/bin/env bash
# Monitor the authorized pilot, certify persisted tables, then submit and certify the bounded remainder.
set -eo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE=${1:?new absolute data-area state directory}
[ "${2:-}" = --approved-remainder ] || { echo 'explicit --approved-remainder required'; exit 2; }
case "$STATE" in /exp/mu2e/data/users/ram2aq/crv_fastsim/*) ;; *) exit 2 ;; esac
[ "$(hostname -s)" = mu2egpvm07 ] || { echo 'reserved host is mu2egpvm07'; exit 2; }
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
muse setup ops
source /exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/site.env
set -euo pipefail
PY=/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python3
AREA=$M2FS_CODE/crv/art
mkdir "$STATE"
exec 9>"$CRVFS_DATA/thread_a_grid_continuation.lock"
flock -n 9 || { echo 'another continuation owns the lock'; exit 2; }
cd "$STATE"
LEDGER=$CRVFS_DATA/thread_a_remainder_105_120_attempts
mkdir -p "$LEDGER"
if compgen -G "$LEDGER/*.attempted" >/dev/null; then
  echo 'a prior remainder submission attempt requires manual reconciliation'; exit 2
fi
export PYTHONPATH="$CODE:$M2FS_CODE/core${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m crvfastsim.analyses.reference_certification --help > certification_cli.txt
trap 'rc=$?; printf "exit=%s utc=%s\n" "$rc" "$(date -u +%FT%TZ)" > "$STATE/driver_exit.txt"; if [ "$rc" != 0 ]; then echo "STOPPED: inspect $STATE; never blindly rerun a submission"; fi' EXIT
printf 'User authorized six pilots and conditional 16/config remainder on 2026-10-25.\n' > authorization.txt
printf 'run1a 94339906 jobsub02.fnal.gov\nrun2 87294168 jobsub01.fnal.gov\n' > pilot_clusters.txt
printf 'phase=initializing\n' > status.txt
BASELINE=$CRVFS_DATA/thread_a_ref50k_20261025/dump_manifest.txt
head -n 8 "$BASELINE" | tail -n 4 > dumper.sha256
sha256sum -c dumper.sha256
sha256sum "$CODE/grid/grid_ref_sim.sh" "$CODE/scripts/dump_named_transport_reference.sh" \
  "$CODE/scripts/propose_named_reference_extension.sh" "$CODE/scripts/continue_named_reference_grid.sh" \
  "$CODE/crvfastsim/analyses/reference_certification.py" > code.sha256
bash "$CODE/scripts/propose_named_reference_extension.sh" remainder > remainder_commands.txt
sha256sum remainder_commands.txt >> code.sha256

check_fcl() {
  local cfg file
  : > "$STATE/fcl_checked.sha256"
  for cfg in run1a run2; do
    for file in ref_sim_extracted.fcl "ref_sim_$cfg.fcl"; do
      cmp "$CODE/grid/fcl/$file" "$CRVFS_PNFS/g4ref_${cfg}_v1/fcl/$file"
      sha256sum "$CODE/grid/fcl/$file" >> "$STATE/fcl_checked.sha256"
    done
  done
  cmp "$CODE/grid/fcl/geom_cosmic_run2_v10.txt" "$CRVFS_PNFS/g4ref_run2_v1/fcl/geom_cosmic_run2_v10.txt"
  sha256sum "$CODE/grid/fcl/geom_cosmic_run2_v10.txt" >> "$STATE/fcl_checked.sha256"
}
check_fcl
cp fcl_checked.sha256 fcl_frozen.sha256

queue() {
  local out=$1
  timeout 180 jobsub_q -G mu2e --user ram2aq > "$out" 2>&1
  awk '
    $1=="JOBSUBJOBID" {header=1; next}
    $1 ~ /^[0-9]+[.][0-9]+@/ {if (NF<9 || $2!="ram2aq" || $6 !~ /^[IRCHXS]$/) bad=1; next}
    NF==0 || $2=="total;" || /^Attempting to get token/ || /^Storing bearer token/ {next}
    {bad=1}
    END {exit (!header || bad)}' "$out" || { cat "$out"; echo 'queue result not recognized'; return 2; }
  cat "$out"
}

inventory() {
  local cfg=$1 out=$2
  timeout 180 ifdh ls "$CRVFS_PNFS/g4ref_${cfg}_v1" > "$out" 2>&1
  timeout 180 ifdh ls "$CRVFS_PNFS/g4ref_${cfg}_v1/logs" >> "$out" 2>&1
}

wait_stage() {
  local phase=$1 first=$2 last=$3 clusters=$4 iteration cfg cluster schedd k seq kind expected missing active
  local deadline=$((SECONDS + 43200))
  mkdir "$STATE/$phase"
  iteration=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    iteration=$((iteration + 1))
    printf 'phase=%s waiting iteration=%s utc=%s\n' "$phase" "$iteration" "$(date -u +%FT%TZ)" | tee "$STATE/status.txt"
    queue "$STATE/$phase/queue.$iteration.txt"
    active=0
    while read -r cfg cluster schedd; do
      if awk -v c="$cluster" -v s="$schedd" 'split($1,a,"@")==2 && a[2]==s && a[1] ~ ("^" c "[.][0-9]+$") {found=1} END {exit !found}' "$STATE/$phase/queue.$iteration.txt"; then
        active=1
        if awk -v c="$cluster" '$1 ~ ("^" c "[.]") && ($6=="H" || $6=="X" || $6=="S") {found=1} END {exit !found}' "$STATE/$phase/queue.$iteration.txt"; then
          echo "held/removed/suspended $cfg jobs require inspection"; return 2
        fi
      fi
    done < "$clusters"
    if [ "$active" = 0 ]; then
      missing=0
      for cfg in run1a run2; do
        inventory "$cfg" "$STATE/$phase/inventory.$cfg.$iteration.txt"
        for k in $(seq "$first" "$last"); do
          seq=$(printf '001400_%08d' "$k")
          for kind in dts gen; do
            expected="$CRVFS_PNFS/g4ref_${cfg}_v1/$kind.$CRVFS_OWNER.CosmicCRY${cfg}Ref.v1.$seq.art"
            grep -Fxq "$expected" "$STATE/$phase/inventory.$cfg.$iteration.txt" || missing=1
          done
          grep -Fxq "$CRVFS_PNFS/g4ref_${cfg}_v1/logs/mu2e.$seq.log.gz" "$STATE/$phase/inventory.$cfg.$iteration.txt" || missing=1
        done
      done
      [ "$missing" = 0 ] && break
      echo 'jobs no longer queued but copy-out evidence incomplete; stopping without resubmission'; return 2
    fi
    sleep 300
  done
  [ "$active" = 0 ] || { echo 'monitor deadline exceeded'; return 2; }
  while read -r cfg cluster schedd; do
    for k in $(seq 0 "$((last-first))"); do
      timeout 180 jobsub_fetchlog -G mu2e --partial -J "$cluster.$k@$schedd" \
        --destdir "$STATE/$phase/gridlogs/$cfg/$k" > "$STATE/$phase/fetch.$cfg.$k.txt" 2>&1
      grep -R -F -x -q "=== grid_ref_sim: done shard $k" "$STATE/$phase/gridlogs/$cfg/$k" || {
        echo "missing worker completion marker for $cfg process $k"; return 2;
      }
    done
  done < "$clusters"
}

certify() {
  local phase=$1 first=$2 last=$3 used quota
  printf 'phase=%s certifying utc=%s\n' "$phase" "$(date -u +%FT%TZ)" | tee "$STATE/status.txt"
  sha256sum -c "$STATE/code.sha256"
  sha256sum -c "$STATE/dumper.sha256"
  used=$(getfattr --only-values -n ceph.dir.rbytes "$M2FS_DATA")
  quota=$(getfattr --only-values -n ceph.quota.max_bytes "$M2FS_DATA")
  printf 'used=%s quota=%s\n' "$used" "$quota" > "$STATE/$phase/quota.txt"
  [ "$((quota-used))" -gt 2147483648 ] || { echo 'less than 2 GiB quota headroom'; return 2; }
  (
    set +u
    cd "$AREA"
    muse setup
    set -euo pipefail
    [ "$MUSE_WORK_DIR" = "$AREA" ] || exit 2
    bash "$CODE/scripts/dump_named_transport_reference.sh" "$STATE/$phase/tables" "$AREA" "$first" "$last"
  ) > "$STATE/$phase/dump.log" 2>&1
  (
    "$PY" -m crvfastsim.analyses.reference_certification "$STATE/$phase/tables" \
      --first "$first" --last "$last" --expected-events 50000 --output "$STATE/$phase/certification.json"
  ) > "$STATE/$phase/certification.log" 2>&1
}

wait_stage pilot 102 104 "$STATE/pilot_clusters.txt"
certify pilot 102 104
printf 'phase=pilot_certified utc=%s\n' "$(date -u +%FT%TZ)" | tee status.txt
sha256sum -c code.sha256
sha256sum -c fcl_frozen.sha256
check_fcl
queue "$STATE/remainder_preflight_queue.txt"
if awk '$1 ~ /^[0-9]+[.][0-9]+@/ {found=1} END {exit !found}' "$STATE/remainder_preflight_queue.txt"; then
  echo 'nonempty user queue: stop for explicit dedup review'; exit 2
fi
for cfg in run1a run2; do
  inventory "$cfg" "$STATE/remainder_preflight_$cfg.txt"
  for k in $(seq 105 120); do
    seq=$(printf '001400_%08d' "$k")
    if grep -Fq "$seq" "$STATE/remainder_preflight_$cfg.txt"; then
      echo "unexpected existing remainder artifact $cfg/$seq"; exit 2
    fi
  done
done
: > remainder_clusters.txt
for cfg in run1a run2; do
  if [ "$cfg" = run1a ]; then base=20264100; else base=20265100; fi
  out=$CRVFS_PNFS/g4ref_${cfg}_v1
  printf 'phase=submitting_remainder config=%s utc=%s\n' "$cfg" "$(date -u +%FT%TZ)" | tee status.txt
  (set -o noclobber; printf 'attempted %s state=%s\n' "$(date -u +%FT%TZ)" "$STATE" > "$LEDGER/$cfg.attempted")
  cp "$LEDGER/$cfg.attempted" "$STATE/remainder_submit_$cfg.attempted"
  jobsub_submit -N 16 -G mu2e --role=Analysis --expected-lifetime=4h \
    --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC \
    --lines=+FERMIHTC_AutoRelease=True --memory=3000MB --disk=4000MB \
    --need-storage-modify="${out/#\/pnfs\/mu2e/\/mu2e}" \
    -e NEVTS=50000 -e BASESEED="$base" -e FIRST_SUBRUN=105 -e TAG=v1 -e CFG="$cfg" \
    -e FCL_PNFS="$out/fcl" -e MUSING=MDC2025ay -e OWNER="$CRVFS_OWNER" \
    "file://$CODE/grid/grid_ref_sim.sh" "$out" > "$STATE/remainder_submit_$cfg.log" 2>&1
  cat "$STATE/remainder_submit_$cfg.log"
  id=$(sed -n 's/.*Use job id \([0-9]*\)\.0@\([^ ]*\) to retrieve.*/\1 \2/p' "$STATE/remainder_submit_$cfg.log")
  [ "$(printf '%s\n' "$id" | wc -w)" -eq 2 ] || { echo 'ambiguous submit outcome: inspect queue'; exit 2; }
  printf '%s %s\n' "$cfg" "$id" >> remainder_clusters.txt
done
wait_stage remainder 105 120 "$STATE/remainder_clusters.txt"
certify remainder 105 120
printf 'phase=COMPLETE new_persisted_generated_per_config=950000 retained_subrun101_per_config=50000 utc=%s\n' "$(date -u +%FT%TZ)" | tee status.txt
