#!/bin/bash
# Stock chain of a reco response build (recoresp.config): the full dts events through the manifest's stock OffSpill
# digitisation, reconstruction and EventNtuple, one sequencer per array task.
# Usage:
#   python3 -m trkfastsim.recoresp.config render MANIFEST FCLDIR      (FCLDIR/{digi,reco,ntuple}.fcl, manifest.json)
#   sbatch -A YOUR_ACCOUNT --array=0-N resp_build_slurm.sh FCLDIR LIST OUT_ROOT
#     LIST: lines "SEQUENCER DTS_PATH"; output OUT_ROOT/<SEQUENCER>/nts.<SEQUENCER>.root
#   python3 -m trkfastsim.recoresp.table --steps-dirs ... --nts OUT_ROOT --pools all --config FCLDIR/manifest.json ...
# Site paths come from the environment: FASTSIMKIT_SCRATCH (scratch root), OFFLINE_TARBALL (muse area),
# OFFLINE_CHAIN_RUNNER (runner with OFFLINE_CHAIN support), OFFLINE_IMAGE, CVMFSEXEC_DIST.
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 04:00:00
set -eo pipefail
module load apptainer
FCLDIR=$(readlink -f "$1"); LIST=$2; O=$(readlink -m "$3")     # resolved: the container binds the real path
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the scratch root}")
read -r SEQ DTS < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
DTS=$(readlink -f "$DTS")
A=${OFFLINE_TARBALL:?muse area tarball}
R=${OFFLINE_CHAIN_RUNNER:?runner with OFFLINE_CHAIN support}
W=$O/$SEQ
mkdir -p "$W/fcl"
cd "$W"
[ -s "$DTS" ]
for f in digi reco ntuple; do cp "$FCLDIR/$f.fcl" fcl/$f.fcl; done
cp "$FCLDIR/manifest.json" manifest.json
[ ! -d "$FCLDIR/files" ] || cp "$FCLDIR"/files/* .          # input files of the manifest (found by name)
cat > chain.sh <<CH
set -eo pipefail
export MU2E_SEARCH_PATH="$W:\$MU2E_SEARCH_PATH"   # input files of the manifest, found by name (DbService text files)
export TMPDIR=\$(dirname "\$OFFLINE_CACHE_PATH")/tmp
export SQLITE_TMPDIR=\$TMPDIR
mkdir -p "\$TMPDIR"
cd "$W"
for f in digi reco ntuple; do fhicl-dump -c fcl/\$f.fcl > fcl/\$f.expanded.fcl; done
"\$OFFLINE_TIME" -v -o digi.time mu2e -c fcl/digi.fcl -s "$DTS" > digi.log 2>&1
"\$OFFLINE_TIME" -v -o reco.time mu2e -c fcl/reco.fcl -s dig.art > reco.log 2>&1
"\$OFFLINE_TIME" -v -o ntuple.time mu2e -c fcl/ntuple.fcl -s mcs-LH.art -T nts.$SEQ.root.tmp > ntuple.log 2>&1
grep -h "TrigReport Events total" digi.log reco.log ntuple.log > events.txt || true
mv nts.$SEQ.root.tmp nts.$SEQ.root
rm -f mcs-LH.art dig.art
CH
export CVMFSEXEC_DIST=${CVMFSEXEC_DIST:-$B/environments/cvmfsexec-el8}
export OFFLINE_IMAGE=${OFFLINE_IMAGE:?pinned EL9 worker image directory on CVMFS}
export OFFLINE_BINDS="$B:$B" OFFLINE_CHAIN="$W/chain.sh" OFFLINE_LOG_PREFIX=chain
bash "$R" "$A" "$W/fcl/reco.fcl" > runner.log 2>&1
