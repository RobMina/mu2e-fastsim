#!/bin/bash
# No-Geant4 fixture and saved-art reread under the already-enforced inner handoff.
set -eo pipefail
[[ $PWD == /mnt/work && $HOME == /mnt/work/home && $TMPDIR == /mnt/work/tmp ]]
A=/opt/r5/area
: "${1:?outer PID namespace required}"
/usr/bin/python3 -I -B /opt/r5/checks/flash_reference_apptainer_probe.py --outer-pid-namespace "$1" --persistent-target /opt/r5/area/r5-fixture-must-not-write --persistent-target /opt/r5/checks/r5-fixture-must-not-write --persistent-target /scratch/ram2aq/r5-fixture-must-not-write
if [[ $(. /etc/os-release; echo "$ID:$VERSION_ID") == rocky:9* ]]; then
  export MU2E_OSNAME=al9 MU2E_SPACK=true UPS_OVERRIDE="-H Linux64bit+5.14-2.34-al9-3"
fi
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
setup muse
cd "$A"
muse setup
[[ $(readlink -f "$MUSE_WORK_DIR") == "$A" ]]
[[ $MUSE_STUB == al9-prof-e29-p109 && $MUSE_ENVSET == p109 ]]
export FHICL_FILE_PATH="$A:${FHICL_FILE_PATH}:$A/build/$MUSE_STUB"
BB="$(readlink -f "$A/backing")/build/$MUSE_STUB"
[[ ! -d $BB ]] || export FHICL_FILE_PATH="$FHICL_FILE_PATH:$BB"
sha256sum "$(command -v mu2e)" "$A/build/$MUSE_STUB/FlashReference/lib/"*.so
mkdir /mnt/work/fixture /mnt/work/reread
cd /mnt/work/fixture
mu2e -c FlashReference/fcl/fixture.fcl -T export.root
[[ -f flash_reference_fixture_1.art && -f export.root ]]
cd /mnt/work/reread
mu2e -c FlashReference/fcl/read_fixture.fcl -s /mnt/work/fixture/flash_reference_fixture_1.art -T export.root
[[ -f export.root ]]
printf 'confined_noG4_fixture_and_saved_reread_exit0=true\n'
wc -c /mnt/work/fixture/flash_reference_fixture_1.art /mnt/work/fixture/export.root /mnt/work/reread/export.root
sha256sum /mnt/work/fixture/flash_reference_fixture_1.art /mnt/work/fixture/export.root /mnt/work/reread/export.root
