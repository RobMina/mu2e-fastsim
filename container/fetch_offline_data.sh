#!/bin/sh
# Fetch the public Offline files the fast chain reads (pinned Offline tag) into <dest>/Offline/...; used by the image
# build and by jobs on sites without CVMFS. usage: fetch_offline_data.sh <dest> [offline_ref]
set -e
D=${1:?usage: fetch_offline_data.sh <dest> [offline_ref]}; REF=${2:-v13_39_00}
U=https://raw.githubusercontent.com/Mu2e/Offline/$REF
for f in CRVResponse/data/singlePEWaveform_v3.txt Mu2eG4/geom/crv_counters_extracted_v04.txt; do
  mkdir -p "$D/Offline/$(dirname $f)"
  [ -s "$D/Offline/$f" ] || curl -fsSL -o "$D/Offline/$f" "$U/$f"
  echo "$REF $f $(wc -l < "$D/Offline/$f") lines"
done
