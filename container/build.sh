#!/bin/bash
# Build the mu2e-fastsim Apptainer image from this umbrella checkout on a machine with apptainer and user namespaces
# (e.g. a Mu2e GPVM). usage: container/build.sh [out.sif]. ~10 min.
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
sed "s#^    \. /opt/mu2e-fastsim#    $STAGE /opt/mu2e-fastsim#" container/mu2efastsim.def > "$STAGE.def"
apptainer build --fakeroot --force "$OUT" "$STAGE.def"
rm -f "$STAGE.def"
ls -la "$OUT"; apptainer test "$OUT"
