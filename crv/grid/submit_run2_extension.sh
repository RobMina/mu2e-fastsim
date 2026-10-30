#!/usr/bin/env bash
# Print a frozen Run 2 phase; --submit additionally requires approval and current live preflight.
set -euo pipefail
STATE=$(cd "${1:?prepared state directory}" && pwd)
PHASE=${2:?test, pilot or remainder}
ACTION=${3:---dry-run}
[[ "$ACTION" = --dry-run || "$ACTION" = --submit ]] || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$(cd "$HERE/.." && pwd)
case "$PHASE" in
  test) FIRST=121; JOBS=1; REQUIRED="" ;;
  pilot) FIRST=122; JOBS=2; REQUIRED="121" ;;
  remainder) FIRST=124; JOBS=377; REQUIRED="121 122 123" ;;
  *) exit 2 ;;
esac
(cd "$STATE" && sha256sum -c campaign.sha256)
(cd "$STATE/payload" && sha256sum -c payload.sha256)
read -r OUT OWNER TAG SCOPE SHA < <(python3 - "$STATE" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'payload/campaign.json'
c = json.loads(p.read_text())
assert c['schema'] == 'crv-run2-grid-extension-v1'
assert (c['first_subrun'], c['last_subrun'], c['events_per_job'], c['baseseed']) == (121, 500, 50000, 20265100)
assert c['musing'] == 'MDC2025ay' and c['config'] == 'run2'
print(c['outdir'], c['owner'], c['tag'], c['write_scope'], hashlib.sha256(p.read_bytes()).hexdigest())
PY
)
[[ -n "$OUT" && -n "$SHA" ]] || exit 2
PAYLOAD="$OUT/config/$SHA"
CMD=(jobsub_submit -N "$JOBS" -G mu2e --role=Analysis --expected-lifetime=4h
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC
  --lines=+FERMIHTC_AutoRelease=True --memory=3000MB --disk=4000MB
  --need-storage-modify="$SCOPE"
  -e NEVTS=50000 -e BASESEED=20265100 -e FIRST_SUBRUN="$FIRST"
  -e OWNER="$OWNER" -e TAG="$TAG" -e CAMPAIGN_SHA256="$SHA" -e PAYLOAD_PNFS="$PAYLOAD"
  "file://$STATE/payload/grid_run2_extension.sh" "$OUT")
printf 'phase=%s jobs=%s subruns=%s..%s seeds=%s..%s output=%s\n' "$PHASE" "$JOBS" "$FIRST" "$((FIRST+JOBS-1))" "$((20265100+FIRST-1))" "$((20265100+FIRST+JOBS-2))" "$OUT"
printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$ACTION" = --submit ]] || { echo 'DRY RUN: no pnfs writes or submission'; exit 0; }
for tool in jobsub_submit jobsub_q ifdh; do command -v "$tool"; done
[[ "$PWD" != "$CODE"* && "$PWD" != /pnfs/* ]] || { echo 'submit from a local state/data directory'; exit 2; }
exec 9> "$STATE/submission.lock"
flock -n 9 || { echo 'another submission owns the lock'; exit 2; }
if [[ -e "$STATE/$PHASE.attempted" ]]; then
  echo 'prior submission attempt: reconcile queue/outputs manually before any new attempt'; exit 2
fi
if [[ -n "$REQUIRED" ]]; then
  PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" python3 - "$STATE/gate_$PHASE.json" "$SHA" $REQUIRED <<'PY'
import sys
from crvfastsim.analyses.run2_grid_plan import validate_gate
validate_gate(sys.argv[1], sys.argv[2], list(map(int, sys.argv[3:])))
PY
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
Q="$STATE/$PHASE.queue.$STAMP.txt"
jobsub_q -G mu2e --user "$OWNER" > "$Q" 2>&1 || { cat "$Q"; exit 2; }
grep -q JOBSUBJOBID "$Q" || { echo 'unrecognized queue snapshot'; cat "$Q"; exit 2; }
if grep -E 'grid_run2_|grid_ref_sim|CosmicCRYrun2Ref' "$Q"; then
  echo 'another reference job is queued; reconcile its identities before submitting'; exit 2
fi
INVENTORY="$STATE/inventory.$STAMP.txt"
: > "$INVENTORY"
for directory in "$OUT" "$OUT/logs" "$OUT/receipts"; do
  ifdh ls "$directory" >> "$INVENTORY" 2>> "$STATE/inventory.$STAMP.err" || {
    echo 'Namespace check failed (possibly absent); review errors before proceeding.'
    cat "$STATE/inventory.$STAMP.err"
    [[ "${RUN2_NEW_NAMESPACE_APPROVED:-}" = "$OUT" ]] || exit 2
  }
done
for subrun in $(seq "$FIRST" "$((FIRST+JOBS-1))"); do
  SEQ=$(printf '001400_%08d' "$subrun")
  if grep -F "$SEQ" "$INVENTORY"; then
    echo "output exists for $SEQ; stop, do not submit duplicates"; exit 2
  fi
done
for directory in "$OUT" "$OUT/logs" "$OUT/receipts" "$PAYLOAD"; do
  ifdh mkdir_p "$directory" || exit 2
done
for file in campaign.json payload.sha256 ref_sim_extracted.fcl ref_sim_run2.fcl geom_cosmic_run2_v10.txt grid_run2_extension.sh; do
  if ifdh ls "$PAYLOAD/$file" >/dev/null 2>&1; then
    CHECK=$(mktemp)
    ifdh cp "$PAYLOAD/$file" "$CHECK" && cmp "$STATE/payload/$file" "$CHECK" || { rm -f "$CHECK"; exit 2; }
    rm -f "$CHECK"
  else
    ifdh cp "$STATE/payload/$file" "$PAYLOAD/$file" || exit 2
  fi
done
(set -o noclobber; printf 'utc=%s phase=%s first=%s jobs=%s\n' "$STAMP" "$PHASE" "$FIRST" "$JOBS" > "$STATE/$PHASE.attempted")
printf '%q ' "${CMD[@]}" > "$STATE/$PHASE.command.txt"; printf '\n' >> "$STATE/$PHASE.command.txt"
set +e
"${CMD[@]}" 2>&1 | tee "$STATE/$PHASE.submit.$STAMP.log"
RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$RC" > "$STATE/$PHASE.submit.exit"
echo 'Keep the attempt marker even on failure; inspect actual queue and outputs before retrying.'
exit "$RC"
