# Shared part of the site presets (site/eaf.env, site/gpvm.env, site/slurm.env); sourced by them, not directly.
# Needs M2FS_INPUTS, MANIFEST and M2FS_DATA; everything else is derived from them, this checkout, $USER and $HOME.

if [ -n "${BASH_SOURCE[0]:-}" ]; then _m2fs_site=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
else _m2fs_site=$(cd "$(dirname "$0")" && pwd); fi
export M2FS_CODE="${M2FS_CODE:-$(cd "$_m2fs_site/.." && pwd)}"
unset _m2fs_site

: "${M2FS_INPUTS:?set M2FS_INPUTS to the release root from your access request (setup/README.md)}"
: "${MANIFEST:?set MANIFEST to the release manifest from your access request (setup/README.md)}"
: "${M2FS_DATA:?set M2FS_DATA to a writable output area of yours (on Fermilab storage: your Mu2e data area)}"
export M2FS_INPUTS MANIFEST M2FS_DATA

export PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker:$M2FS_CODE/beam${PYTHONPATH:+:$PYTHONPATH}"
export CRVFS_CODE="$M2FS_CODE/crv" TRKFS_CODE="$M2FS_CODE/tracker" M2FS_CORE="$M2FS_CODE/core"
export CRVFS_DATA="$M2FS_INPUTS/crv_fastsim"
export CRVFS_LOOKUP_DIR="$CRVFS_DATA/lookup_v6_0"
export MU2E_SEARCH_PATH="$M2FS_INPUTS/offline"
export CRVFS_OFFLINE_CACHE="$M2FS_DATA/cache/offline"
export BEAMFS_DATA_ROOT="$M2FS_DATA/beam_fastsim"
export M2FS_VENV="${M2FS_VENV:-$M2FS_DATA/venv_fastsim}"
# inputs the kit reads in place (EAF) or streams from the FNAL host (Slurm), as seen on Fermilab storage
export FASTSIMKIT_FNAL_DATA="$CRVFS_DATA" FASTSIMKIT_CRV_DATA="$CRVFS_DATA"
export FASTSIMKIT_BEAM_DATA="$M2FS_INPUTS/beam_fastsim" FASTSIMKIT_TRACKER_DATA="$M2FS_INPUTS"
export FASTSIMKIT_GEOMETRY_PREFIX="$M2FS_INPUTS/"
export FASTSIMKIT_STATE_DIR="${FASTSIMKIT_STATE_DIR:-$HOME/.mu2efastsim/kit}"
