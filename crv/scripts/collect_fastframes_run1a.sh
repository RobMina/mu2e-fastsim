#!/bin/bash
# Collect the F-fast grid outputs (submit_fastframes_run1a.sh) and compare the three frame sets on the same events:
# fast vs production (EventNtuple crvcoincs), fast vs stock redigi on the kept steps, redigi vs production; plus F
# deadtime with the dev card on each. Seconds of CPU: runs on a GPVM.
#   collect_fastframes_run1a.sh [OUT]   (default $CRVFS_DATA/deadtime_run1a/ffast_grid)
D=${CRVFS_DATA:-/exp/mu2e/data/users/ram2aq/crv_fastsim}; C=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
G=/pnfs/mu2e/scratch/users/ram2aq/crvfs_fastframes_run1a; O=${1:-$D/deadtime_run1a/ffast_grid}; mkdir -p $O/in || exit 2
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup ops >/dev/null 2>&1 || true
for f in $(ifdh ls $G/out | grep -E "frames_fast_|crvredigi_" | xargs -n1 basename); do
  [ -s $O/in/$f ] || ifdh cp $G/out/$f $O/in/$f >/dev/null 2>&1 || echo "(copy failed: $f)"
done
[ -s $O/in/list.txt ] || ifdh cp $G/in/list.txt $O/in/list.txt >/dev/null 2>&1
ls $O/in/frames_fast_*.parquet | sed 's/.*frames_fast_//; s/.parquet//' > $O/subruns.txt
NTS=$(grep -F -f $O/subruns.txt $O/in/list.txt | awk '{print $4}')
echo "subruns: $(wc -l < $O/subruns.txt)"
PY=/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python3   # explicit: pyenv ana after muse setup ops can leave another python first
cd $C && export PYTHONPATH=core:crv
$PY - "$O" <<'PY'
import sys, awkward as ak, glob
from crvfastsim.veto.compareframes import dump_frames
o = sys.argv[1]
for f in sorted(glob.glob(f"{o}/in/crvredigi_*.root")):
    s = f.split("_")[-1].replace(".root", "")
    t = f"{o}/in/frames_redigi_{s}.parquet"
    ak.to_parquet(dump_frames([f]), t)
PY
CUTS=$D/mu2e_cosmic_ana/config/common/cuts.yaml; SEL=$D/deadtime_run1a/ce_mix1bb_dev_sel.parquet
$PY -m crvfastsim.veto.compareframes $O/in/frames_fast_*.parquet --nts $NTS --selection $SEL --cosmic-ana $CUTS dev --outdir $O/fast_vs_prod > $O/fast_vs_prod.log 2>&1
$PY -m crvfastsim.veto.compareframes $O/in/frames_redigi_*.parquet --nts $NTS --selection $SEL --cosmic-ana $CUTS dev --outdir $O/redigi_vs_prod > $O/redigi_vs_prod.log 2>&1
$PY -m crvfastsim.veto.compareframes $O/in/frames_fast_*.parquet --dump $O/in/crvredigi_*.root --selection $SEL --cosmic-ana $CUTS dev --outdir $O/fast_vs_redigi > $O/fast_vs_redigi.log 2>&1
$PY - "$O" <<'PY'
import json, sys
o = sys.argv[1]
for c in ("fast_vs_prod", "fast_vs_redigi", "redigi_vs_prod"):
    j = json.load(open(f"{o}/{c}/compare_frames.json")); m = j.get("matched_single", {})
    a, b = c.split("_vs_")
    print(f"{c:15s} frames {j['common_frames']}: {a} {j['fast']['coincs']} ({j['fast']['frac_frames_with_coinc']:.4f}), "
          f"{b} {j['stock']['coincs']} ({j['stock']['frac_frames_with_coinc']:.4f}); both/{a}-only/{b}-only "
          f"{j['both']}/{j['fast_only']}/{j['stock_only']}; PE KS {j['ks_PEs']:.3f}, t KS {j['ks_time']:.3f}; "
          f"matched {m.get('n')}: PE ratio {m.get('pe_ratio_median', float('nan')):.3f}, dt {m.get('dt_median', float('nan')):+.2f} ns; "
          f"F {a} {100 * j['F_fast']['frac']:.3f} +- {100 * j['F_fast']['err']:.3f} %, F {b} {100 * j['F_stock_same_frames']['frac']:.3f} +- {100 * j['F_stock_same_frames']['err']:.3f} %")
PY
