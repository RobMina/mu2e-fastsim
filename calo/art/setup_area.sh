#!/bin/bash
# Prepare isolated backing links and an explicit envset; never run setup or a build.
set -eo pipefail
: "${CALO_BACKING:?absolute pinned backing path required}"
: "${CALO_ENVSET:?explicit matching envset required}"
AREA=$(cd "$(dirname "$0")" && pwd)
[[ "$CALO_BACKING" = /* ]] || exit 1
[[ "$CALO_ENVSET" =~ ^p[0-9]+$ ]] || exit 1
test -d "$CALO_BACKING/build"
test -d "$CALO_BACKING/build/al9-prof-e29-$CALO_ENVSET"
test -f "$CALO_BACKING/Offline/.muse"
grep -Eq "^ENVSET[[:space:]]+$CALO_ENVSET[[:space:]]*$" "$CALO_BACKING/Offline/.muse"
test -d "$AREA/CaloFastSim"
test ! -L "$AREA/CaloFastSim"
test -f "$AREA/CaloFastSim/src/SConscript"
if test -e "$AREA/backing" || test -L "$AREA/backing"; then
  test "$(readlink "$AREA/backing")" = "$CALO_BACKING"
else
  ln -s "$CALO_BACKING" "$AREA/backing"
fi
if test -e "$AREA/.muse"; then
  test "$(cat "$AREA/.muse")" = "ENVSET $CALO_ENVSET"
else
  printf 'ENVSET %s\n' "$CALO_ENVSET" > "$AREA/.muse"
fi
printf 'Prepared area: %s\nBacking: %s\nEnvset: %s\n' "$AREA" "$CALO_BACKING" "$CALO_ENVSET"
printf 'User only, on the designated build host:\n'
printf 'cd %q && source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && setup muse && muse setup && muse build -j 32\n' "$AREA"
