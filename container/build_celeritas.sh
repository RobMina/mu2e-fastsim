#!/bin/bash
# Build the Celeritas image (container/celeritas.def) on a UVA node, by the user (rule G6). Long: spack builds Geant4,
# VecGeom with CUDA and Celeritas from source (expect 1-3 h on 32 cores). Needs apptainer with --fakeroot and outbound
# internet (UVA compute nodes have both).
# usage: [CELERITAS=0.6.3] [SPACK_VERSION=v1.2.2] [JOBS=32] [SQUASHFS_MEM=4G] [SQUASHFS_PROCS=n] container/build_celeritas.sh [out.sif]
set -e
cd "$(dirname "$0")"
OUT=$(realpath -m "${1:-mu2e-celeritas.sif}")
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}/$USER/apptainer_tmp}; mkdir -p "$APPTAINER_TMPDIR"
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${TMPDIR:-/tmp}/$USER/apptainer_cache}
DEF=$(mktemp "$APPTAINER_TMPDIR/celeritas.XXXX.def")
sed -e "s#@CELERITAS@#${CELERITAS:-0.6.3}#g" -e "s#@SPACK_VERSION@#${SPACK_VERSION:-v1.2.2}#g" -e "s#@JOBS@#${JOBS:-32}#g" \
  celeritas.def > "$DEF"
# mksquashfs defaults to 25 % of the node's RAM and all node CPUs, not the job's: cap both
MEM=${SQUASHFS_MEM:-4G}
PROCS=${SQUASHFS_PROCS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
apptainer build --fakeroot --force --mksquashfs-args "-mem $MEM -processors $PROCS" "$OUT" "$DEF"
rm -f "$DEF"
ls -la "$OUT"; apptainer test "$OUT"
