#!/bin/bash
# Build the mu2e-fastsim Apptainer image from this umbrella checkout on a machine with apptainer and user namespaces
# (a Mu2e GPVM, or a Polaris compute node). usage: [VARIANT=gpu|cpu] [TRIM="..."] container/build.sh [out.sif]
# gpu (default): JAX + pip CUDA 12 wheels without TRIM (default: the cuDNN, NCCL, NVSHMEM and cuFFT wheels, which no
# kernel loads); TRIM="" keeps them all. cpu: CPU-only JAX (FermiGrid, local, CI). ~2-10 min.
# Only git-tracked files of the umbrella and its submodules go in (a `git archive` export at the checked-out commits):
# never site.env, work areas, build products or Mu2e-internal data.
set -e
cd "$(dirname "$0")/.."
OUT=$(realpath -m "${1:-mu2efastsim.sif}")
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/tmp/$USER/apptainer_tmp}; mkdir -p "$APPTAINER_TMPDIR"
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/tmp/$USER/apptainer_cache}
STAGE=$(mktemp -d "$APPTAINER_TMPDIR/stage.XXXX"); trap 'rm -rf "$STAGE"' EXIT
git archive HEAD | tar -x -C "$STAGE"
for p in core crv tracker; do
  [ -f $p/pyproject.toml ] || { echo "submodule $p missing"; exit 2; }
  mkdir -p "$STAGE/$p" && git -C $p archive HEAD | tar -x -C "$STAGE/$p"
done
{ git log -1 --format="umbrella %H"; git submodule status; } > "$STAGE/VERSION"
cat "$STAGE/VERSION"; du -sh "$STAGE"
VARIANT=${VARIANT:-gpu}
TRIM=${TRIM-nvidia-cudnn-cu12 nvidia-nccl-cu12 nvidia-nvshmem-cu12 nvidia-cufft-cu12}
[ "$VARIANT" = cpu ] && TRIM=""
echo "variant $VARIANT trim [$TRIM]" | tee -a "$STAGE/VERSION"
sed -e "s#^    \. /opt/mu2e-fastsim#    $STAGE /opt/mu2e-fastsim#" -e "s#@VARIANT@#$VARIANT#g" -e "s#@TRIM@#$TRIM#g" \
  container/mu2efastsim.def > "$STAGE.def"
apptainer build --fakeroot --force "$OUT" "$STAGE.def"
rm -f "$STAGE.def"
ls -la "$OUT"; apptainer test "$OUT"
