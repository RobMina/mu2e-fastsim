#!/bin/bash
# Stream to an already allocated node; aliases must reuse existing authenticated masters without dialing.
set -euo pipefail
SITE=${1:?polaris or aurora}; JID=${2:?existing PBS job id}
CRV=${3:?local isolated CRV clone}; CORE=${4:?local isolated core clone}
case "$SITE" in
    polaris) F=/eagle/GenesisHackathonOct26/ram2aq/crvfastsim ;;
    aurora) F=/lus/flare/projects/GenesisHackathonOct26/ram2aq/mu2efastsim ;;
    *) exit 2 ;;
esac
case "$JID" in ''|*[!0-9]*) echo 'Numeric PBS job id required'; exit 2;; esac
U=$SITE; G=mu2eaigpvm01; D=/exp/mu2e/data/users/ram2aq/crv_fastsim
O=$F/bench/agent13_$JID
LOCAL=$(mktemp -d "${TMPDIR:-/tmp}/agent13_alcf_stream_XXXXXX")
trap 'rm -rf -- "$LOCAL"' EXIT
alcf() { ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=/usr/bin/false "$U" "$@"; }
while ! INBOX=$(alcf "cat $O/inbox" </dev/null 2>/dev/null); do
    alcf "qstat $JID" </dev/null || { echo 'Allocation not queued or running'; exit 3; }
    sleep 15
done
read -r HOST L REMOTE_OUT <<< "$INBOX"
test "$REMOTE_OUT" = "$O"
case "$HOST" in ''|*[!a-zA-Z0-9.-]*) echo 'Invalid node'; exit 3;; esac
case "$L" in /tmp/crv_agent13_${JID}_*|/dev/shm/crv_agent13_${JID}_*) ;; *) echo "Unapproved tmpfs path: $L"; exit 3;; esac
case "$L" in *[!a-zA-Z0-9_./-]*) exit 3;; esac
printf 'node=%s tmp=%s out=%s\n' "$HOST" "$L" "$O"
test "$(dirname "$CRV")" = "$(dirname "$CORE")"
COPYFILE_DISABLE=1 tar -czf "$LOCAL/code.tgz" --exclude .git --exclude __pycache__ --exclude .pytest_cache \
    -s ",^$(basename "$CRV"),crv," -s ",^$(basename "$CORE"),core," \
    -C "$(dirname "$CRV")" "$(basename "$CRV")" "$(basename "$CORE")"
put() {
    local name=$1 expected=$2 size=$3 got
    if test "$SITE" = polaris; then
        got=$(alcf "PORT=\$(cat $O/upload.port); curl --fail --silent --show-error --noproxy '*' --cacert $O/upload.crt --connect-to agent13-upload:\$PORT:$HOST:\$PORT -H @$O/upload.headers -H 'Content-Length: $size' -H 'Transfer-Encoding:' --upload-file - https://agent13-upload:\$PORT/$name")
    else
        got=$(alcf "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new $HOST 'umask 077; cat > $L/$name && sha256sum $L/$name'")
    fi
    test "${got%% *}" = "$expected" || { echo "Checksum mismatch: $name"; exit 4; }
    if test "$name" != manifest.sha256; then printf '%s  %s\n' "$expected" "$name" >> "$LOCAL/manifest.sha256"; fi
    echo "verified $name"
}
put code.tgz "$(shasum -a 256 "$LOCAL/code.tgz" | cut -d' ' -f1)" "$(wc -c < "$LOCAL/code.tgz" | tr -d ' ')" < "$LOCAL/code.tgz"
for rel in g4ref_v1_tables/gen.001400_00000011.root tier1b_v7/lib_prod_as_20files.root \
    geometry/crvcalib_sim_best_v1_5_run1400.csv geometry/crvphoton_sim_best_v1_5_run1400.csv geometry/mu2e_extracted_v04.gdml; do
    src=$D/$rel
    sum=$(ssh -o BatchMode=yes "$G" "sha256sum $src" </dev/null)
    size=$(ssh -o BatchMode=yes "$G" "wc -c < $src" </dev/null)
    ssh -o BatchMode=yes "$G" "cat $src" </dev/null | put "$(basename "$src")" "${sum%% *}" "$size"
done
if test "$SITE" = polaris; then
    put manifest.sha256 "$(shasum -a 256 "$LOCAL/manifest.sha256" | cut -d' ' -f1)" "$(wc -c < "$LOCAL/manifest.sha256" | tr -d ' ')" < "$LOCAL/manifest.sha256"
    alcf "PORT=\$(cat $O/upload.port); curl --fail --silent --show-error --noproxy '*' --cacert $O/upload.crt --connect-to agent13-upload:\$PORT:$HOST:\$PORT -H @$O/upload.headers -X POST https://agent13-upload:\$PORT/ready" </dev/null
else
    alcf "ssh -o BatchMode=yes $HOST 'umask 077; cat > $L/manifest.sha256 && cd $L && sha256sum -c manifest.sha256 && touch ready'" < "$LOCAL/manifest.sha256"
fi
echo "Stream complete: site=$SITE job=$JID output=$O"
