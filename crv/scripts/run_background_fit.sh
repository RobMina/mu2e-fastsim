#!/bin/bash
# run_background_fit.sh OUTDIR DATA_LIST MC_GLOB HOST [HOST...] -- fit the out-of-time background model (reco.background)
# as a GPVM scatter: one data file per partial (~30 s per 500 k-event file, 1.4 GB), each host taking its share
# serially (one job per host), then the merge on the first host. Launch from the laptop (it holds the ssh masters).
# Outputs (mode 700 dir): OUTDIR/parts/{data_NN,mc_NN}.npz, OUTDIR/bg.npz, OUTDIR/background_model.json, logs/.
set -u
OUT=${1:?outdir}; LIST=${2:?data file list}; MCG=${3:?mc glob}; shift 3
HOSTS=("$@"); [ ${#HOSTS[@]} -gt 0 ] || { echo "need hosts"; exit 2; }
CODE=${M2FS_CODE:?set M2FS_CODE to the mu2e-fastsim checkout on the hosts}
ENV="source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1; pyenv ana >/dev/null 2>&1; export PYTHONPATH=$CODE/core:$CODE/crv"
ssh "${HOSTS[0]}" "mkdir -p $OUT/parts $OUT/logs && chmod 700 $OUT" </dev/null
NH=${#HOSTS[@]}
for i in "${!HOSTS[@]}"; do
  ssh "${HOSTS[$i]}" "nohup bash -c '$ENV; n=0; while read -r f; do [ -n \"\$f\" ] || continue; k=\$n; n=\$((n+1)); [ \$((k % $NH)) -eq $i ] || continue; P=$OUT/parts/data_\$(printf %02d \$k).npz; [ -s \$P ] || python3 -m crvfastsim reco-background \"\$f\" --emit \$P --seed \$((20261022 + k)) >> $OUT/logs/data_\$k.log 2>&1; done < $LIST; echo DONE > $OUT/logs/host_$i.done' > /dev/null 2>&1 &" </dev/null
done
ssh "${HOSTS[0]}" "nohup bash -c '$ENV; k=0; for f in $MCG; do P=$OUT/parts/mc_\$(printf %02d \$k).npz; [ -s \$P ] || python3 -m crvfastsim reco-background \"\$f\" --emit \$P --rate-only >> $OUT/logs/mc_\$k.log 2>&1; k=\$((k+1)); done; echo DONE > $OUT/logs/mc.done' > /dev/null 2>&1 &" </dev/null
echo "launched on ${HOSTS[*]}; wait for $OUT/logs/host_*.done and mc.done, then:"
echo "  ssh ${HOSTS[0]} \"$ENV; python3 -m crvfastsim reco-background $OUT/parts/data_*.npz --mc $OUT/parts/mc_*.npz --merge --out $OUT/bg.npz --outdir $OUT\""
