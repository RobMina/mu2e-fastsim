#!/usr/bin/env bash
# Frozen Run 2 reference extension; all production outputs and completion evidence go to pnfs.
set -eo pipefail
OUTDIR=${1:?pnfs output directory}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
muse setup SimJob MDC2025ay || true
setup ifdhc 2>/dev/null || true
set -u
command -v mu2e
command -v ifdh
: "${PROCESS:?}" "${FIRST_SUBRUN:?}" "${NEVTS:?}" "${BASESEED:?}" "${OWNER:?}" "${TAG:?}" "${PAYLOAD_PNFS:?}" "${CAMPAIGN_SHA256:?}"
[[ "$PROCESS" =~ ^[0-9]+$ && "$FIRST_SUBRUN" =~ ^[0-9]+$ ]] || exit 2
SUBRUN=$((FIRST_SUBRUN + PROCESS))
SEED=$((BASESEED + SUBRUN - 1))
[[ "$SUBRUN" -ge 121 && "$SUBRUN" -le 500 && "$NEVTS" = 50000 && "$BASESEED" = 20265100 ]] || exit 2
SEQ=$(printf '001400_%08d' "$SUBRUN")
DTS="dts.$OWNER.CosmicCRYrun2Ref.$TAG.$SEQ.art"
GEN="gen.$OWNER.CosmicCRYrun2Ref.$TAG.$SEQ.art"
RECEIPT="complete.$SEQ.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p fcl
for file in campaign.json payload.sha256 ref_sim_extracted.fcl ref_sim_run2.fcl geom_cosmic_run2_v10.txt grid_run2_extension.sh; do
  ifdh cp "$PAYLOAD_PNFS/$file" "fcl/$file" || exit 2
done
printf '%s  %s\n' "$CAMPAIGN_SHA256" fcl/campaign.json | sha256sum -c -
(cd fcl && sha256sum -c payload.sha256) || exit 2
python3 - "$OUTDIR" "$SUBRUN" "$SEED" <<'PY'
import hashlib, json, os, pathlib, sys
c = json.load(open('fcl/campaign.json'))
assert c['outdir'] == sys.argv[1] and c['owner'] == os.environ['OWNER'] and c['tag'] == os.environ['TAG']
assert c['events_per_job'] == int(os.environ['NEVTS']) and c['musing'] == 'MDC2025ay'
assert {'subrun': int(sys.argv[2]), 'seed': int(sys.argv[3]), 'generated_events': c['events_per_job']} in c['shards']
for name, expected in c['payload_sha256'].items():
    assert pathlib.Path(name).name == name
    assert hashlib.sha256(pathlib.Path('fcl', name).read_bytes()).hexdigest() == expected
PY
cmp "$0" fcl/grid_run2_extension.sh || { echo 'worker differs from frozen payload'; exit 2; }
: > remote_inventory.txt
for directory in "$OUTDIR" "$OUTDIR/logs" "$OUTDIR/receipts"; do
  ifdh ls "$directory" >> remote_inventory.txt || { echo 'cannot verify output namespace'; exit 2; }
done
exists() { grep -Fxq "$1" remote_inventory.txt; }
if exists "$OUTDIR/receipts/$RECEIPT"; then
  ifdh cp "$OUTDIR/receipts/$RECEIPT" existing.json || exit 2
  python3 - "$CAMPAIGN_SHA256" "$SUBRUN" "$SEED" "$NEVTS" <<'PY'
import json, sys
r = json.load(open('existing.json'))
assert r['campaign_sha256'] == sys.argv[1]
assert [r['subrun'], r['seed'], r['generated_events']] == list(map(int, sys.argv[2:]))
assert r['status'] == 'complete' and r['copyout_complete']
PY
  exists "$OUTDIR/$DTS" && exists "$OUTDIR/$GEN" && exists "$OUTDIR/logs/mu2e.$SEQ.log.gz" || exit 2
  echo "SKIP: complete receipt and both art outputs exist for $SEQ"
  exit 0
fi
if exists "$OUTDIR/$DTS" || exists "$OUTDIR/$GEN" || exists "$OUTDIR/logs/mu2e.$SEQ.log.gz"; then
  echo "ERROR: partial unreceipted output; reconcile $SEQ, never overwrite"
  exit 2
fi
cat > fcl/shard.fcl <<EOF
#include "ref_sim_run2.fcl"
source.firstSubRun : $SUBRUN
services.SeedService.baseSeed : $SEED
outputs.PrimaryOutput.fileName : "$DTS"
outputs.GenOutput.fileName : "$GEN"
EOF
export FHICL_FILE_PATH="$PWD/fcl:$FHICL_FILE_PATH"
export MU2E_SEARCH_PATH="$PWD/fcl:$MU2E_SEARCH_PATH"
echo "Run2 extension subrun=$SUBRUN seed=$SEED events=$NEVTS host=$(hostname) utc=$(date -u +%FT%TZ)"
mu2e --version > runtime.txt 2>&1 || exit 3
python3 - <<'PY'
import os
stub = os.environ.get('MUSE_STUB', '')
work = os.path.realpath(os.environ.get('MUSE_WORK_DIR', ''))
assert stub == 'al9-prof-e29-p107', 'unexpected runtime stub: ' + stub
assert work.endswith('/Musings/SimJob/MDC2025ay'), 'unexpected Musing: ' + work
backing = os.path.realpath(os.path.join(work, 'backing'))
assert backing.endswith('/Musings/Offline/v13_39_00'), 'unexpected Offline backing: ' + backing
PY
mu2e --debug-config fcl/resolved.fcl -c fcl/shard.fcl --annotate > expand.log 2>&1 || { tail -60 expand.log; exit 3; }
/usr/bin/time -v mu2e -c fcl/shard.fcl -n "$NEVTS" > mu2e.log 2>&1 || { tail -60 mu2e.log; exit 3; }
grep -E 'TrigReport Events total|Elapsed|Maximum resident|Exit status' mu2e.log
python3 - "$NEVTS" <<'PY'
import re, sys
text = open('mu2e.log').read()
rows = re.findall(r'TrigReport Events total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)', text)
assert len(rows) == 1, 'missing or ambiguous job event summary'
n, p, f = map(int, rows[0])
assert n == int(sys.argv[1]) and p + f == n
assert 'Exit status: 0' in text
PY
sha256sum "$DTS" "$GEN" fcl/* runtime.txt > outputs.sha256
stat -c '%n %s' "$DTS" "$GEN" > output_bytes.txt
gzip -c mu2e.log > "mu2e.$SEQ.log.gz"
tar -czf "provenance.$SEQ.tar.gz" fcl runtime.txt outputs.sha256 output_bytes.txt
ifdh cp "$DTS" "$OUTDIR/$DTS" || exit 5
ifdh cp "$GEN" "$OUTDIR/$GEN" || exit 5
ifdh cp "mu2e.$SEQ.log.gz" "$OUTDIR/logs/mu2e.$SEQ.log.gz" || exit 5
ifdh cp "provenance.$SEQ.tar.gz" "$OUTDIR/receipts/provenance.$SEQ.tar.gz" || exit 5
python3 - "$CAMPAIGN_SHA256" "$SUBRUN" "$SEED" "$NEVTS" "$DTS" "$GEN" <<'PY'
import hashlib, json, os, re, sys
text = open('mu2e.log').read()
n, p, f = map(int, re.findall(r'TrigReport Events total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)', text)[0])
def artifact(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'name': path, 'bytes': os.path.getsize(path), 'sha256': h.hexdigest()}
r = {'schema': 'crv-run2-grid-completion-v1', 'status': 'complete', 'copyout_complete': True,
     'campaign_sha256': sys.argv[1], 'subrun': int(sys.argv[2]), 'seed': int(sys.argv[3]),
     'generated_events': n, 'accepted_events_log': p, 'failed_events_log': f,
     'persisted_event_counts_verified': False, 'musing': 'MDC2025ay',
     'artifacts': [artifact(path) for path in sys.argv[5:]]}
with open('complete.json', 'x') as out:
    json.dump(r, out, indent=2)
PY
ifdh cp complete.json "$OUTDIR/receipts/$RECEIPT" || exit 5
echo "DONE: $SEQ published with receipt; persisted-count certification still required"
