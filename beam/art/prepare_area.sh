#!/bin/bash
# Create only an explicit backing link; never build or change an existing area's backing.
set -eo pipefail
[[ $# == 1 ]] || { echo 'usage: prepare_area.sh EXISTING_OFFLINE_BACKING' >&2; exit 2; }
AREA=$(cd "$(dirname "$0")" && pwd -P)
BACKING=$(readlink -f "$1")
[[ -d "$BACKING/Offline" && -d "$BACKING/build" ]] || { echo 'Backing must be a complete existing Offline area' >&2; exit 2; }
if [[ -e "$AREA/backing" || -L "$AREA/backing" ]]; then
  [[ -L "$AREA/backing" && $(readlink -f "$AREA/backing") == "$BACKING" ]] || { echo 'Existing backing differs; refusing replacement' >&2; exit 2; }
else
  ln -s "$BACKING" "$AREA/backing"
fi
printf 'area=%s\nbacking=%s\nsource_prepared_only=true\n' "$AREA" "$BACKING"
