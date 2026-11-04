#!/bin/sh
# Fetch the CRY 1.7 sea-level table the device CRY source reads (cosmics_0.data) into <dest>/data, with the CRY
# licence notices into <dest>/share/licenses; used by the image build. The source archive, the table and the
# additional notice are checked against pinned SHA256s; nothing is compiled. usage: fetch_cry_data.sh <dest>
set -e
D=${1:?usage: fetch_cry_data.sh <dest>}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
curl -fsSL --retry 3 --max-time 180 -o "$T/cry_v1.7.tar.gz" https://nuclear.llnl.gov/simulation/cry_v1.7.tar.gz
echo "dcee2428f81cba113f82e0c7c42f4d85bff4b8530e5ab5c82c059bed3e570c20  $T/cry_v1.7.tar.gz" | sha256sum -c -
tar -xzf "$T/cry_v1.7.tar.gz" -C "$T" cry_v1.7/data/cosmics_0.data cry_v1.7/src/COPYRIGHT.TXT
mkdir -p "$D/data" "$D/share/licenses"
cp "$T/cry_v1.7/data/cosmics_0.data" "$D/data/cosmics_0.data"
echo "b5227d8c1d38ab249a3b238456a67a60c9a9a5b6f91f748562f4a517390dca75  $D/data/cosmics_0.data" | sha256sum -c -
cp "$T/cry_v1.7/src/COPYRIGHT.TXT" "$D/share/licenses/CRY-COPYRIGHT.TXT"
curl -fsSL --retry 3 --max-time 60 -o "$D/share/licenses/CRY-additional_bsd.html" https://nuclear.llnl.gov/simulations/additional_bsd.html
echo "732e701b1f9c3bfc8b8d57b53ad116cdde1367aef1942ea50796a7d6ea0dbc9b  $D/share/licenses/CRY-additional_bsd.html" | sha256sum -c -
(cd "$D/data" && sha256sum cosmics_0.data > "$D/DATA.sha256")
printf '%s\n' "CRY 1.7 sea-level table (cosmics_0.data) from https://nuclear.llnl.gov/simulation/cry_v1.7.tar.gz" \
  "third-party: BSD-style licence and an additional LLNL/DOE notice, $D/share/licenses/" > "$D/NOTICE"
