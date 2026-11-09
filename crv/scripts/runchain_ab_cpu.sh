#!/bin/bash
# runchain on CPU with two crv checkouts (same core, seed, inputs): checks a refactor leaves the extracted chain
# unchanged. usage: runchain_ab_cpu.sh <crv_dir_A> <crv_dir_B> <gen.root> [max_events=10000]
# needs pyenv ana and CRVFS_DATA; outputs $CRVFS_DATA/runcfg_dev/ab_{A,B}; prints both cluster tables
A=${1:?}; B=${2:?}; GEN=${3:?}; N=${4:-10000}; D=${CRVFS_DATA:?}; CORE=${M2FS_CODE:?set M2FS_CODE to the mu2e-fastsim checkout}/core
for arm in A B; do
  dir=${!arm}; O=$D/runcfg_dev/ab_$arm; mkdir -p "$O"
  ( cd "$dir" && PYTHONPATH=$CORE:$dir python3 -m crvfastsim runchain "$GEN" --library "$D/tier1b_v7/lib_prod_as_20files.root" \
      --calib "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv" --deviation "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" \
      --max-events "$N" --seed 7 --outdir "$O" > "$O/runchain.log" 2>&1 ); echo "$arm rc=$?"
done
python3 - "$D/runcfg_dev/ab_A/runchain_stats.json" "$D/runcfg_dev/ab_B/runchain_stats.json" <<'EOF'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
for k in ("steps", "events_with_steps", "photons", "charges", "digis", "pulses"):
    print(k, a["timers"].get(k), b["timers"].get(k))
print("clusters", a["clusters_per_generated"], b["clusters_per_generated"])
print("identical:", all(a["timers"].get(k) == b["timers"].get(k) for k in ("steps", "photons", "charges", "digis", "pulses"))
      and a["clusters_per_generated"] == b["clusters_per_generated"])
EOF
