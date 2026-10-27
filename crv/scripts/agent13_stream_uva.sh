#!/bin/bash
# Stream owned source and private inputs to an already submitted allocation, never to a mirrored tree.
set -euo pipefail
JID=${1:?existing Slurm job id}
CRV=${2:?local isolated CRV clone}
CORE=${3:?local isolated core clone}
U=uva-hpc
G=mu2eaigpvm01
D=/exp/mu2e/data/users/ram2aq/crv_fastsim
O=/scratch/ram2aq/mu2efastsim/probe/agent13_$JID
LOCAL=$(mktemp -d "${TMPDIR:-/tmp}/agent13_stream_XXXXXX")
trap 'rm -rf -- "$LOCAL"' EXIT
while ! INBOX=$(ssh "$U" "cat $O/inbox" </dev/null 2>/dev/null); do
    state=$(ssh "$U" "squeue -h -j $JID -o %T" </dev/null)
    test -n "$state" || { echo 'Allocation is not queued or running'; exit 3; }
    sleep 10
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
    name=$1
    expected=$2
    got=$(ssh "$U" "srun --jobid=$JID --overlap -N1 -n1 --mem=0 -w $HOST sh -c 'cat > $L/$name && sha256sum $L/$name'")
    test "${got%% *}" = "$expected" || { echo "Checksum mismatch: $name"; exit 4; }
    printf '%s  %s\n' "$expected" "$name" >> "$LOCAL/manifest.sha256"
    echo "verified $name"
}
put code.tgz "$(shasum -a 256 "$LOCAL/code.tgz" | cut -d' ' -f1)" < "$LOCAL/code.tgz"
for rel in g4ref_v1_tables/gen.001400_00000011.root g4ref_v1_tables/gen.001400_00000012.root \
    g4ref_v1_tables/gen.001400_00000013.root tier1b_v7/lib_prod_as_20files.root \
    geometry/crvcalib_sim_best_v1_5_run1400.csv geometry/crvphoton_sim_best_v1_5_run1400.csv geometry/mu2e_extracted_v04.gdml; do
    src=$D/$rel
    sum=$(ssh "$G" "sha256sum $src" </dev/null)
    ssh "$G" "cat $src" </dev/null | put "$(basename "$src")" "${sum%% *}"
done
ssh "$U" "srun --jobid=$JID --overlap -N1 -n1 --mem=0 -w $HOST sh -c 'cat > $L/manifest.sha256 && cd $L && sha256sum -c manifest.sha256 && touch ready'" < "$LOCAL/manifest.sha256"
echo "Stream complete: job=$JID output=$O"
