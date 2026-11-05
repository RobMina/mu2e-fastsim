#!/bin/bash
# No transport: validate unchanged plugins, extraction and private namespace features.
set -eo pipefail
: "${R5_PROBE_ROOT:?}" "${R5_PROBE_AREA:?}" "${R5_PROBE_CODE:?}" "${R5_HOST_BWRAP:?}"
test "$MUSE_WORK_DIR" = "$R5_PROBE_AREA"
cd "$R5_PROBE_ROOT"
printf 'MUSE_WORK_DIR=%s\nMUSE_ENVSET=%s\nMUSE_STUB=%s\n' "$MUSE_WORK_DIR" "$MUSE_ENVSET" "$MUSE_STUB" > environment.txt
mu2e --version >> environment.txt
sha256sum "$(command -v mu2e)" > executable.sha256
for plugin in "$R5_PROBE_AREA"/build/al9-prof-e29-p109/FlashReference/lib/*.so; do
  sha256sum "$plugin" >> plugins.sha256
  ldd -r "$plugin" >> libraries.txt 2>&1
done
if grep -E 'not found|undefined symbol' libraries.txt; then exit 91; fi
mu2e --print-description FlashReferenceExporter > exporter.description.txt
mu2e --print-description FlashReferenceFixture > fixture.description.txt
mkdir fixture reread source first10 remaining990
(cd fixture && mu2e -c FlashReference/fcl/fixture.fcl -T fixture.root > read.log 2>&1)
(cd reread && mu2e -c FlashReference/fcl/read_fixture.fcl -s "$R5_PROBE_ROOT/fixture/flash_reference_fixture_1.art" -T reread.root > read.log 2>&1)
INPUT=/scratch/ram2aq/mu2efastsim/flash_r5_20261103/inputs/elebeamcat_ab_20000.art
printf '%s  %s\n' a0932e89ea3e2a94a16dcbde62fa81543ce0486e263c540293851b9fa67cd5f4 "$INPUT" | sha256sum -c - > source_hash.txt
(cd source && mu2e -c FlashReference/fcl/read_source_1000.fcl -s "$INPUT" -T source.root > read.log 2>&1)
(cd first10 && mu2e -c FlashReference/fcl/read_source_first10.fcl -s "$INPUT" -T source.root > read.log 2>&1)
(cd remaining990 && mu2e -c FlashReference/fcl/read_source_remaining990.fcl -s "$INPUT" -T source.root > read.log 2>&1)
A=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_41_01
sha256sum "$A/Offline/Mu2eG4/geom/bfgeom_v01.txt" "$A/Offline/GeometryService/src/BFieldManagerMaker.cc" > field_config.sha256
for name in DSMap PSMap TSuMap_fix TSdMap PStoDumpAreaMap ProtonDumpAreaMap DSExtension PSAreaMap WorldMap; do
  file=/cvmfs/mu2e.opensciencegrid.org/DataFiles/BFieldMaps/Mau13/$name
  sha256sum "$file.header" "$file.bin" >> field_payloads.sha256
done
"$R5_HOST_BWRAP" --version > bwrap.txt
"$R5_HOST_BWRAP" --unshare-user --unshare-pid --unshare-ipc --unshare-uts --die-with-parent --new-session --ro-bind / / --proc /proc --dev /dev /bin/true
printf 'fixture_and_reread_passed=true\ntransport_events=0\ncompleted=%s\n' "$(date -Is)" > complete.txt
