#!/bin/bash
# Collect completed summary evidence to a new non-mirrored local directory before removing the remote copy.
set -euo pipefail
SITE=${1:?polaris or aurora}; JID=${2:?numeric PBS job id}; DEST=${3:?new non-mirrored local directory}
case "$SITE" in
    polaris) F=/eagle/GenesisHackathonOct26/ram2aq/crvfastsim ;;
    aurora) F=/lus/flare/projects/GenesisHackathonOct26/ram2aq/mu2efastsim ;;
    *) exit 2 ;;
esac
case "$JID" in ''|*[!0-9]*) exit 2;; esac
case "$DEST" in */remote/*|*/remote-data/*|*/remote-uva*/*|*/goose-sync/*) echo 'Use a non-mirrored destination'; exit 2;; esac
O=$F/bench/agent13_$JID
alcf() { ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=/usr/bin/false "$SITE" "$@"; }
alcf "test -f $O/exit_code && test -f $O/cleanup.txt && test ! -f $O/upload.token && test ! -f $O/upload.headers" </dev/null
umask 077
mkdir "$DEST"
alcf "cd $O && find . -type f ! -name summary.sha256 -print0 | sort -z | xargs -0 sha256sum > summary.sha256 && tar -cf - ." </dev/null | tar -xf - -C "$DEST"
(cd "$DEST" && shasum -a 256 -c summary.sha256)
alcf "cd $O && sha256sum -c summary.sha256 >/dev/null && cd / && rm -rf -- $O" </dev/null
printf 'Verified summaries retained at %s; removed %s:%s\n' "$DEST" "$SITE" "$O"
