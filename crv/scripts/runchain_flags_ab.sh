#!/bin/bash
# runchain twice on the same inputs and seed, without and with extra flags (e.g. --prefetch): prints counts and whether
# they are identical. usage: runchain_flags_ab.sh "<flags>" <gen.root> [<gen.root> ...]   (CPU, extracted, 5k events/file)
# needs pyenv ana, CRVFS_DATA, PYTHONPATH (core + this crv); outputs $CRVFS_DATA/runcfg_dev/flags_{base,cand}
FL=${1?flags}; shift; D=${CRVFS_DATA:?}
for arm in base cand; do
  O=$D/runcfg_dev/flags_$arm; mkdir -p "$O"; X=""; [ $arm = cand ] && X=$FL
  python3 -m crvfastsim runchain "$@" --library "$D/tier1b_v7/lib_prod_as_20files.root" \
    --calib "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv" --deviation "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" \
    --max-events 5000 --seed 3 $X --outdir "$O" > "$O/runchain.log" 2>&1; echo "$arm rc=$?"
done
python3 - "$D/runcfg_dev/flags_base/runchain_stats.json" "$D/runcfg_dev/flags_cand/runchain_stats.json" <<'EOF'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
ks = ("steps", "events_with_steps", "photons", "charges", "digis", "pulses")
for k in ks:
    print(k, a["timers"].get(k), b["timers"].get(k))
print("clusters", a["clusters_per_generated"], b["clusters_per_generated"], "wall", round(a["wall_s"], 1), round(b["wall_s"], 1))
print("identical:", all(a["timers"].get(k) == b["timers"].get(k) for k in ks) and a["clusters_per_generated"] == b["clusters_per_generated"])
EOF
