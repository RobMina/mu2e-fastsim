#!/bin/bash
# vr5_collect.sh OUTDIR [DATAREF] -- V-R5: collect both arms' objective accumulators on the GPVM and score every point.
#   stock: ifdh cp $VR5_BASE/{vr5_v1,vr5_v1ws}/emit/<point>/*.json; nominal = crvfastsim.objective on the crvonly_v05
#          production ntuples of the same five files (identical code and FHiCL) unless already emitted by the grid test
#   fast:  OUTDIR/fast_raw/<pbsjob>/{knobs.txt,r3.gpu<N>.json}, copied from the Eagle outdirs beforehand (the GPVM cannot
#          reach ALCF: from the laptop, ssh polaris "tar -C <outdir> -cf - ." | ssh <gpvm> "tar -C .../fast_raw/<job> -xf -");
#          split per point by knobs.txt; nominal seeds kept apart (nominal_s<job>). FAST_EMPTY: events without CRV steps
#          to add to each fast point whose JSON predates r3's own count (V-R5 v1: 125,433 - 112,304 = 13,129)
# Env VR5_RUNS: stock runs to collect (default "vr5_v1 vr5_v1ws"; light propagation: fiberprop_v1).
# Output: OUTDIR/{stock,fast}/<point>.json (merged accumulator) and OUTDIR/scores.json + scores.txt (objective vs DATAREF
# per point per arm). Run on a GPVM after `source setupmu2e-art.sh && pyenv ana` with PYTHONPATH=<mu2e-fastsim>/crv.
set -u
OUT=${1:?outdir}; REF=${2:-${CRVFS_DATA:?source site.env}/r3/dataref/dataref_3runs.json}
RUNS=${VR5_RUNS:-vr5_v1 vr5_v1ws}                   # stock runs under $VR5_BASE to collect (e.g. fiberprop_v1)
P=${VR5_BASE:?set VR5_BASE to the dCache scratch directory of the V-R5 runs}; V05=${CRVONLY_V05_NTS:?set CRVONLY_V05_NTS to the crvonly_v05 ntuple directory}
SEQS="00000990 00001023 00001056 00001089 00001122"
mkdir -p "$OUT/stock_emit" "$OUT/stock" "$OUT/fast"
for run in $RUNS; do
  for pt in $(ifdh ls $P/$run/emit 2>/dev/null | sed -n "s|^$P/$run/emit/\([^/]*\)/\?$|\1|p"); do
    mkdir -p "$OUT/stock_emit/$pt"
    for f in $(ifdh ls $P/$run/emit/$pt 2>/dev/null | grep '\.json$'); do
      [ -s "$OUT/stock_emit/$pt/$(basename $f)" ] || ifdh cp "$f" "$OUT/stock_emit/$pt/$(basename $f)" >/dev/null 2>&1
    done
  done
done
mkdir -p "$OUT/stock_emit/nominal"
for q in $SEQS; do
  J="$OUT/stock_emit/nominal/001400_$q.json"; [ -s "$J" ] && continue
  T=$(mktemp -d -t vr5n_XXXX)
  ifdh cp "$V05/nts.crvonly05.001400_$q.root" "$T/n.root" >/dev/null 2>&1 && python3 -m crvfastsim.objective "$T/n.root" --emit "$J" >/dev/null
  rm -rf "$T"
done
for d in "$OUT"/stock_emit/*/; do
  pt=$(basename "$d"); n=$(ls "$d"/*.json 2>/dev/null | wc -l)
  [ "$n" -eq 5 ] || echo "WARN stock $pt has $n of 5 files"
  python3 -c "import json,glob,sys
from crvfastsim.objective import crvresponse as R
json.dump(R.merge_states([json.load(open(f)) for f in sorted(glob.glob('$d/*.json'))]), open('$OUT/stock/$pt.json','w'))"
done
for fd in "$OUT"/fast_raw/*/; do
  [ -s "$fd/knobs.txt" ] || continue
  job=$(basename "$fd"); g=0
  while IFS= read -r k || [ -n "$k" ]; do
    [ -n "$k" ] && pt=$(echo "$k" | tr -d ' ') || pt="nominal_s$job"
    python3 -c "import json
from crvfastsim.objective import crvresponse as R
j = json.load(open('$fd/r3.gpu$g.json')); S = j['response_state']
e = j.get('events_without_steps')                  # r3 from crv 2026-10-22 on adds them itself
if e is None:
    S = R.add_empty_events(S, int('${FAST_EMPTY:-0}'), 1e5)
json.dump(S, open('$OUT/fast/$pt.json', 'w'))"
    g=$((g + 1))
  done < "$fd/knobs.txt"
done
python3 - "$OUT" "$REF" <<'EOF'
import glob, json, os, sys
from crvfastsim.objective import crvresponse as R
out, ref = sys.argv[1], json.load(open(sys.argv[2]))
res = {}
for arm in ("stock", "fast"):
    for f in sorted(glob.glob(f"{out}/{arm}/*.json")):
        pt = os.path.basename(f)[:-5]
        S = json.load(open(f))
        o = R.objective(S, ref)
        oot = {s: S["oot_n"][s] / max(S["exposure_ns"] * 1e-3, 1e-300) for s in R.SLOTS}
        res.setdefault(pt, {})[arm] = {"objective": o["objective"], "sigma": o["sigma"], "events": S["events"],
                                      "oot_rate": oot, "terms": {k: v["chi2ndf"] for k, v in o["terms"].items()}}
json.dump(res, open(f"{out}/scores.json", "w"), indent=1)
with open(f"{out}/scores.txt", "w") as fh:
    fh.write(f"{'point':22s} {'stock':>16s} {'fast':>16s}   oot EX rate stock / fast\n")
    for pt in sorted(res):
        r = res[pt]
        c = lambda a: f"{r[a]['objective']:8.2f} +- {r[a]['sigma']:4.2f}" if a in r else " " * 16
        e = lambda a: f"{r[a]['oot_rate']['EX']:.4f}" if a in r else "  -   "
        fh.write(f"{pt:22s} {c('stock')} {c('fast')}   {e('stock')} / {e('fast')}\n")
print(open(f"{out}/scores.txt").read())
EOF
