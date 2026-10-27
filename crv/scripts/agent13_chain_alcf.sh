#!/bin/bash
# Submit on an existing authenticated alias with -v AGENT13_SITE=polaris|aurora and the site's filesystems resource.
#PBS -N crv_agent13
#PBS -A GenesisHackathonOct26
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:55:00
#PBS -j oe
set -eo pipefail
umask 077
ulimit -c 0
case ${AGENT13_SITE:?} in
    polaris) F=/eagle/GenesisHackathonOct26/$USER/crvfastsim ;;
    aurora) F=/lus/flare/projects/GenesisHackathonOct26/$USER/mu2efastsim ;;
    *) exit 2 ;;
esac
O=$F/bench/agent13_${PBS_JOBID%%.*}
mkdir -p "$F/bench"
mkdir "$O"
exec > "$O/run.log" 2>&1
if test "$(stat -f -c %T /tmp)" = tmpfs; then TMPBASE=/tmp; else TMPBASE=/dev/shm; fi
test "$(stat -f -c %T "$TMPBASE")" = tmpfs
L=$(mktemp -d "$TMPBASE/crv_agent13_${PBS_JOBID%%.*}_XXXXXX")
stat -f -c %T "$L" > "$O/filesystem.txt"
cleanup() {
    rc=$?
    trap '' TERM INT HUP PIPE
    if test -n "${PAYLOAD_PID:-}"; then kill -- -"$PAYLOAD_PID" 2>/dev/null || true; wait "$PAYLOAD_PID" 2>/dev/null || true; fi
    if test -n "${UPLOAD_PID:-}"; then kill "$UPLOAD_PID" 2>/dev/null || true; wait "$UPLOAD_PID" 2>/dev/null || true; fi
    rm -rf -- "$L"
    rm -f -- "$O/inbox" "$O/upload.token" "$O/upload.headers" "$O/upload.port"
    printf '%s exit=%s deleted=%s\n' "$(date -Is)" "$rc" "$L" >> "$O/cleanup.txt"
    printf '%s\n' "$rc" > "$O/exit_code"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP PIPE
printf 'site=%s host=%s job=%s output=%s\n' "$AGENT13_SITE" "$(hostname)" "$PBS_JOBID" "$O"
lscpu > "$O/cpu.txt"
if test "$AGENT13_SITE" = polaris; then
    test -f "${AGENT13_RECEIVER:?}"
    "$F/venv/bin/python" "$AGENT13_RECEIVER" "$L" "$O" > "$O/upload.log" 2>&1 &
    UPLOAD_PID=$!
    for i in $(seq 30); do test -f "$O/upload.port" && break; sleep 1; done
    test -f "$O/upload.port"
fi
printf '%s %s %s\n' "$(hostname)" "$L" "$O" > "$O/inbox"
t0=$SECONDS
for i in $(seq 360); do test -f "$L/ready" && break; sleep 5; done
printf 'input_wait_s=%s\n' "$((SECONDS-t0))" > "$O/staging.txt"
test -f "$L/ready" || { echo 'No complete input manifest arrived'; exit 3; }
if test -n "${UPLOAD_PID:-}"; then
    kill "$UPLOAD_PID"; wait "$UPLOAD_PID" 2>/dev/null || true; UPLOAD_PID=
    rm -f -- "$O/upload.token" "$O/upload.headers" "$O/upload.port" "$L/upload.key"
fi
(cd "$L" && sha256sum -c manifest.sha256)
cp "$L/manifest.sha256" "$O/manifest.sha256"
mkdir "$L/code"
tar -xzf "$L/code.tgz" -C "$L/code"
export PYTHONPATH=$L/code/crv:$L/code/core MU2E_SEARCH_PATH=$F/offline_data
export CRVFS_LOOKUP_DIR=$L CRVFS_DATA=$L JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
if test "$AGENT13_SITE" = aurora; then
    export LD_LIBRARY_PATH=$F/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
else
    export CUDA_VISIBLE_DEVICES=0
    nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$O/gpu.txt"
fi
V=$F/venv/bin/python
"$V" -c 'import jax, crvfastsim, mu2efastsim; print(jax.__version__, jax.devices(), crvfastsim.__file__, mu2efastsim.__file__); assert jax.default_backend() == "gpu"; assert len(jax.devices()) == 1' > "$O/device.txt" 2>&1
for f in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
    "$V" -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org "/DataFiles/CRVConditions/v6_0/$f" "$L/$f"
done
(cd "$L" && sha256sum LookupTable_* photonMap.root) > "$O/optics.sha256"
setsid taskset -c 1-8 bash "$L/code/crv/scripts/agent13_alcf_payload.sh" "$L" "$O" "$V" "$AGENT13_SITE" &
PAYLOAD_PID=$!
wait "$PAYLOAD_PID"
PAYLOAD_PID=
