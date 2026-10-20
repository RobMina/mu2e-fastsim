#!/bin/bash
# V-R4 with the JAX chain inside the CPU container (FNAL GPVM: inputs read in place, nothing leaves FNAL disks).
# usage: run_vr4_sif.sh FIRST LAST [OUTTAG] [SIF]. Code from this checkout (PYTHONPATH), JAX from the image.
C=/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim; D=/exp/mu2e/data/users/ram2aq/crv_fastsim
A=${1:?first}; B=${2:?last}; TAG=${3:-vr4_jax_v1}; S=${4:-/exp/mu2e/data/users/ram2aq/container/mu2efastsim_cpu.sif}
O=$D/$TAG; mkdir -p $O/batches
X="apptainer exec --cleanenv -B /cvmfs -B /exp/mu2e/app -B /exp/mu2e/data --env PYTHONPATH=$C/core:$C/crv
   --env CRVFS_LOOKUP_DIR=/cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0 --env MU2E_SEARCH_PATH=/opt/offline_data $S"
for k in $(seq $A $B); do
  K=$(printf "001400_%08d" $k)
  [ -s $O/batches/vr4.$K.json ] && continue
  $X python -m crvfastsim reco-vr4 $D/g4ref_v1_tables/crvsteps.$K.root --refs $D/v2_fastC_v1/ref.$K.root \
    --calib $D/geometry/crvcalib_sim_best_v1_5_run1400.csv --deviation $D/geometry/crvphoton_sim_best_v1_5_run1400.csv \
    --jax --seed $k --emit $O/batches/vr4.$K.json --outdir $O > $O/batches/vr4.$K.txt 2>&1
  echo "$K rc=$? $(date +%T)"
done
$X python -m crvfastsim reco-vr4 "$O/batches/vr4.*.json" --merge --outdir $O > $O/vr4_merged.txt 2>&1
tail -8 $O/vr4_merged.txt; echo RUN_VR4_DONE
