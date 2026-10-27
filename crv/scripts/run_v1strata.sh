#!/usr/bin/env bash
# V1 strata diagnostic (crvfastsim.analyses.v1strata) on a named-reference campaign: one shard per subrun, then merge.
# usage: run_v1strata.sh CONFIG(run1a|run2) CAMPAIGN_DIR CANDIDATE_ROOT OUTDIR [SUBRUNS...]
#   CAMPAIGN_DIR: named-reference campaign output (manifest.json lists the certified gen/G4 tables per subrun)
#   CANDIDATE_ROOT: directory with <subrun>/tables/fastC.001400_<subrun>.root (or 'campaign' for the campaign's own)
#   env: CORE (mu2efastsim core checkout, required), GDML (optional: adds the below-plane x/X0 strata), PY,
#        PATH_MAP=FROM=TO (rewrite the manifest's path prefix FROM to TO), V1STATS (merged v1compare_stats.json of
#        the same reference: adds the event-clustered binned-ratio shapes), BELOW_MARGIN (mm, with GDML)
set -eo pipefail
CFG=${1:?run1a or run2}; CAMP=${2:?campaign dir}; CROOT=${3:?candidate root or campaign}; OUT=${4:?outdir}
shift 4
SUBRUNS=${*:-$(seq 101 120)}
[ -e /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh ] && { source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true; }
PY=${PY:-/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python3}
PATH_MAP=${PATH_MAP:-}
set -u
case "$CFG" in run1a) RC=run1a_v01 ;; run2) RC=run2_v10 ;; *) exit 2 ;; esac
CODE=$(cd "$(dirname "$0")/.." && pwd)
CORE=${CORE:?path of the mu2efastsim core checkout}
export PYTHONPATH="$CORE:$CODE"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
for s in $SUBRUNS; do
  read -r GEN REF RETAINED < <("$PY" - "$CAMP/manifest.json" "$CFG" "$s" "$PATH_MAP" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])); cfg, s = sys.argv[2], int(sys.argv[3])
src, _, dst = sys.argv[4].partition("=")
fix = (lambda x: dst + x[len(src):] if src and x.startswith(src) else x)
sh = [x for x in m["shards"] if x["config"] == cfg and x["subrun"] == s][0]
print(fix(sh["artifacts"]["gen"]["path"]), fix(sh["artifacts"]["crvsteps"]["path"]), fix(m["retained"][cfg]["candidate"]))
EOF
)
  if [ "$CROOT" = campaign ]; then
    if [ "$s" = 101 ]; then CAND=$RETAINED; else CAND=$CAMP/$CFG/$s/tables/fastC.001400_$(printf %08d "$s").root; fi
  else
    CAND=$CROOT/$s/tables/fastC.001400_$(printf %08d "$s").root
  fi
  [ -e "$OUT/shard_$s.json" ] && { echo "skip $s (done)"; continue; }
  "$PY" -m crvfastsim.analyses.v1strata --gen "$GEN" --reference "$REF" --candidate "$CAND" --config "$RC" --outdir "$OUT" ${GDML:+--gdml "$GDML"} ${V1STATS:+--v1-stats "$V1STATS"} ${BELOW_MARGIN:+--below-margin "$BELOW_MARGIN"}
done
"$PY" -m crvfastsim.analyses.v1strata --merge "$OUT/shard_*.json" --outdir "$OUT/merged" > "$OUT/merged.txt"
echo "done: $OUT/merged/summary.json"
