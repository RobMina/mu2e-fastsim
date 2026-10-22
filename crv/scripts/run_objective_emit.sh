#!/bin/bash
# run_objective_emit.sh OUTDIR LIST HOST [HOST...] -- crvfastsim.objective accumulators (crvresponse + dtp) of the EventNtuples
# in LIST as a GPVM scatter: the files are dealt round-robin to the hosts, each host runs its share serially (one job per
# host; ~30 s per 500 k-event data file). Launch from the laptop. Env TIME_CORR=<timecal JSON> applies a module time
# correction before accumulating. Outputs OUTDIR/<file stem>.json, OUTDIR/logs/, OUTDIR/logs/host_<i>.done.
set -u
OUT=${1:?outdir}; LIST=${2:?file list}; shift 2; HOSTS=("$@"); [ ${#HOSTS[@]} -gt 0 ] || { echo "need hosts"; exit 2; }
CODE=/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim
ENV="source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1; pyenv ana >/dev/null 2>&1; export PYTHONPATH=$CODE/core:$CODE/crv"
TC=""; [ -n "${TIME_CORR:-}" ] && TC="--time-corr $TIME_CORR"
ssh "${HOSTS[0]}" "mkdir -p $OUT/logs" </dev/null
NH=${#HOSTS[@]}
for i in "${!HOSTS[@]}"; do
  ssh "${HOSTS[$i]}" "nohup bash -c '$ENV; n=0; while read -r f; do [ -n \"\$f\" ] || continue; k=\$n; n=\$((n+1)); [ \$((k % $NH)) -eq $i ] || continue; J=$OUT/\$(basename \"\$f\" .root).json; [ -s \$J ] || python3 -m crvfastsim.objective \"\$f\" --emit \$J $TC >> $OUT/logs/\$k.log 2>&1; done < $LIST; echo DONE > $OUT/logs/host_$i.done' > /dev/null 2>&1 &" </dev/null
done
echo "launched on ${HOSTS[*]} -> $OUT (wait for logs/host_*.done)"
