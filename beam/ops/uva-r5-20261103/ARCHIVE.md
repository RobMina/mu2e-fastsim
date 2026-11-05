# UVA R5 operational source archive — 2026-11-05

This is a source-only copy of the existing mirrored `r5_flash_20261103` directory,
which has no independent Git repository on UVA. It is preserved here for recovery,
not deployed from here and not a portable or approved production entry point.
No job is submitted by archiving these files.

The named `code_*`, `apptainer_boundary_v1` and `exporter_runtime_v1/checks`
directories are **historical experiment snapshots**. They intentionally differ
from the current `beam/beamfastsim/` implementations. In particular, the old pidfd
controller snapshots cannot run on the observed host path and must not be used
as the new pidfd-free controller. The latter has never been host-launched.
Old scripts may reflect superseded resource envelopes, layouts or runtime modes;
read [the current pause handoff](../../docs/r5-pause-20261105.md) before reuse.

Only `.py`, `.sh`, `.sbatch`, `.fcl` and `.md` source files are copied. Plugin
binaries, ROOT/art/NPZ data, caches, build products, bytecode and temporary trees
are excluded. `exporter_runtime_v1/README.md` records the expected user-built
plugin hashes; its referenced `.plugin` files are **not included here**. No binary
build or muse tarball is implied. Site-specific paths/account names in these
operational snapshots are historical provenance, not library defaults.

Raw evidence remains on the original sites and may expire (especially UVA
scratch and node-local work); Git does not back up those artifacts. Unknown-
lifecycle work must not be deleted or signalled merely because a scheduler job
closed. The retained receipts/status descriptions are observations, not process
cleanup or physics-completeness certification.
