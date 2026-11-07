#!/bin/bash
# Prepare the isolated board-accessor area without setup, compilation or execution.
set -eo pipefail
[[ $# == 1 ]] || { echo 'usage: prepare_board_area.sh EXISTING_P107_OFFLINE_BACKING' >&2; exit 2; }
AREA=$(cd "$(dirname "$0")" && pwd -P)
[[ -f "$AREA/FlashBoard/.muse" && -f "$AREA/FlashBoard/src/SConscript" ]] || {
  echo 'FlashBoard requires its repository .muse marker and src/SConscript' >&2; exit 2;
}
BACKING=$(readlink -f "$1")
[[ -d "$BACKING/Offline" && -d "$BACKING/build/al9-prof-e29-p107" ]] || {
  echo 'Backing must provide Offline and the al9-prof-e29-p107 build' >&2; exit 2;
}
grep -Eq '^ENVSET[[:space:]]+p107([[:space:]]|$)' "$BACKING/Offline/.muse" || {
  echo 'Backing environment must match the pinned p107 extraction area' >&2; exit 2;
}
if [[ -e "$AREA/backing" || -L "$AREA/backing" ]]; then
  [[ -L "$AREA/backing" && $(readlink -f "$AREA/backing") == "$BACKING" ]] || {
    echo 'Existing backing differs; refusing replacement' >&2; exit 2;
  }
else
  ln -s "$BACKING" "$AREA/backing"
fi
printf 'area=%s\nbacking=%s\nsource_prepared_only=true\n' "$AREA" "$BACKING"
