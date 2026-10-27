#!/usr/bin/env bash
# Fast transport candidate on a certified named-reference campaign, then V1 (emit per shard, merge) and v1strata.
# usage: run_named_candidate.sh CONFIG(run1a|run2) CAMPAIGN_DIR OUTDIR [SUBRUNS...]
#   CAMPAIGN_DIR: named-reference campaign output (manifest.json lists the certified gen/G4 tables per subrun)
#   env: CORE (mu2efastsim core checkout, required); V1CODE (crv checkout whose v1compare gives the verdict,
#        default this one); TRANSPORT_ARGS (extra transport flags, e.g. --below-margin 2000); PY; GDML;
#        PATH_MAP=FROM=TO rewrites the manifest's path prefix FROM to TO (a local copy of the campaign inputs)
# Seeds follow the campaign: seed = subrun - 100, one input per transport call. Outputs: OUTDIR/<subrun>/tables,
# OUTDIR/<subrun>/v1_emit.json, OUTDIR/v1 (merged V1), OUTDIR/strata (v1strata), logs beside them.
set -eo pipefail
CFG=${1:?run1a or run2}; CAMP=${2:?campaign dir}; OUT=${3:?outdir}
shift 3
SUBRUNS=${*:-$(seq 101 120)}
[ -e /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh ] && { source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true; }
PY=${PY:-/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python3}
PATH_MAP=${PATH_MAP:-}
set -u
case "$CFG" in run1a) RC=run1a_v01 ;; run2) RC=run2_v10 ;; *) exit 2 ;; esac
CODE=$(cd "$(dirname "$0")/.." && pwd)
CORE=${CORE:?path of the mu2efastsim core checkout}
V1CODE=${V1CODE:-$CODE}
GDML=${GDML:-/exp/mu2e/data/users/ram2aq/crv_fastsim/geometry/mu2e_${RC}.gdml}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
SHA=$(command -v sha256sum || echo "shasum -a 256")
$SHA "$CODE/crvfastsim/transport.py" "$CORE/mu2efastsim/transport.py" "$V1CODE/crvfastsim/analyses/v1compare.py" \
  > "$OUT/source_sha256_$(date -u +%Y%m%dT%H%M%SZ).txt"
for s in $SUBRUNS; do
  read -r GEN REF < <("$PY" - "$CAMP/manifest.json" "$CFG" "$s" "$PATH_MAP" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])); cfg, s = sys.argv[2], int(sys.argv[3])
src, _, dst = sys.argv[4].partition("=")
fix = (lambda x: dst + x[len(src):] if src and x.startswith(src) else x)
sh = [x for x in m["shards"] if x["config"] == cfg and x["subrun"] == s][0]
print(fix(sh["artifacts"]["gen"]["path"]), fix(sh["artifacts"]["crvsteps"]["path"]))
EOF
)
  D=$OUT/$s
  CAND=$D/tables/fastC.001400_$(printf %08d "$s").root
  mkdir -p "$D/tables"
  if [ ! -s "$CAND" ]; then
    PYTHONPATH="$CORE:$CODE" "$PY" -m crvfastsim transport "$GEN" --outdir "$D/tables" --seed $((s - 100)) \
      --stage C --run-config "$RC" --gdml "$GDML" ${TRANSPORT_ARGS:-} > "$D/transport.log" 2>&1
  fi
  if [ ! -s "$D/v1_emit.json" ]; then
    PYTHONPATH="$CORE:$V1CODE" "$PY" -m crvfastsim v1compare "$CAND" --compare "$REF" --emit "$D/v1_emit.json" \
      --run-config "$RC" --outdir "$D" > "$D/v1_emit.log" 2>&1
  fi
  echo "subrun $s done $(date -u +%FT%TZ)"
done
rm -rf "$OUT/v1"
rc=0
PYTHONPATH="$CORE:$V1CODE" "$PY" -m crvfastsim v1compare "$OUT/*/v1_emit.json" --merge --run-config "$RC" \
  --outdir "$OUT/v1" > "$OUT/v1.log" 2>&1 || rc=$?
echo "v1 merge exit $rc (1 = FAIL verdict)"
CORE=$CORE GDML=$GDML PY=$PY PATH_MAP=$PATH_MAP V1STATS=$OUT/v1/v1compare_stats.json BELOW_MARGIN=${BELOW_MARGIN:-} bash "$CODE/scripts/run_v1strata.sh" "$CFG" "$CAMP" "$OUT" "$OUT/strata" $SUBRUNS
echo "done: $OUT"
