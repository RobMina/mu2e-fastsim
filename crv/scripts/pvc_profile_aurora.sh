#!/bin/bash
#PBS -N pvcreco_profile
# ALCF project: pass qsub -A "$ALCF_PROJECT" (see site.env.example).
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:flare
#PBS -j oe
set -eo pipefail
umask 077
ulimit -c 0
F=${M2FS_FLARE:?set M2FS_FLARE to the mu2efastsim area on Flare (offline_data/, venv/)}
S=${PVC_RUN_DIR:?set PVC_RUN_DIR to a writable run directory}
O=$S/profile_${PBS_JOBID%%.*}
mkdir "$O"
exec > "$O/run.log" 2>&1
export PYTHONPATH=$S/code/crv:$S/code/core MU2E_SEARCH_PATH=$F/offline_data
export LD_LIBRARY_PATH=$F/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 CRVFASTSIM_COMPACT_DISPATCH=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
V=$F/venv/bin/python
printf 'host=%s job=%s start=%s\n' "$(hostname)" "$PBS_JOBID" "$(date -Is)"
cp "$0" "$O/job.sh"
"$V" -m pip list > "$O/packages.txt"
for backend in cpu oneapi; do
    extra=(); test "$backend" = oneapi && extra=(--against "$O/cpu")
    JAX_PLATFORMS=$backend taskset -c 2-9 "$V" -m crvfastsim.reco.pvc_profile \
        --out "$O/$backend" --rows 4096 --events 1000 --groups 128 --trace \
        "${extra[@]}" > "$O/$backend.log" 2>&1 || exit $?
done
printf 'end=%s\n' "$(date -Is)"
