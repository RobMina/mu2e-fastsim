# Validation sequence for optional recording patches

Review gates, not a submission script or authorization to build/transport. R6 is paused at the 2026-11-05
checkpoint; `../PAUSE_RESUME.md` is the current completion/blocker map. The sequence below remains the acceptance
plan, not a command to repeat completed builds or controls.

## A. Source and scope review

1. Name a separate Offline work area at current canonical head and confirm the user's commit location. Preserve
   the existing saved-step control area and its plugin sources/binaries. Apply no patch to a shared/main checkout.
2. Rebase both draft patches explicitly and retain exact base/patched file hashes. Do not relax hash guards to
   force applicability. Confirm the chosen head, envset, Geant4 version and no unintended tracked-file deletions.
3. Review all included native roots, external cable placement gates, replica/parameterization paths and transforms.
   Existing name inventory is only a candidate. Decide exterior transport ownership and the feedback interface.
4. Inspect mirrored CMake/SConscript dependencies. Current drafts use existing libraries only, with a header-only
   boundary helper; header install/discovery still requires build review. Source-only pytest cannot prove C++ API
   validity, FHiCL types, native touchable availability or unchanged physics.

## B. Build gate

The user runs the normal build on the designated build host after source review. For this checkpoint the exact-CI-
backed three-package overlay (CaloMC, CaloRecording, Mu2eG4) has already been user-built and verified; a full Offline
rebuild is not required. No automatic build script is provided. After any future source change completion must be
explicit; verify actual plugins/linked main libraries, dependencies, freshness, exact config and enabled-path APIs
before any job. Rebuilding only the existing exporter satellite cannot supply Mu2eG4/CaloMC patches in a backing
library. Committing an exact source checkpoint alone does not alter the native source bytes or authorize a rebuild.

## C. Native geometry-only evidence

Before an event campaign, provide an initialized native geometry inventory with stable placement paths/copy
numbers, solid types/parameters, material IDs, parent/child transforms and versioned configuration. Trace every
candidate included volume to its selected root exactly once. A volume-name match or coordinate-envelope test is
not a navigation certificate. Require containment/overlap evidence and explicit treatment of replicas,
parameterizations, radial/phi faces, touching surfaces and disabled/version-dependent cable structures.

Known source-derived probes include radial entry into inter-disk cable material without a disk/FEB VD crossing,
front/back entries, inner/outer cylindrical surfaces, sector-phi faces, adjoining subtree transitions and world
exit. Do not introduce sensitive shells or material-bearing mothers just to simplify observation.

## D. Same-saved-raw compression test before new transport

After the approved build, the existing raw-art pilot can exercise diagnostics off/on **without repeating Geant4**.
Require exact event/product/registry identity, ordered CSS fields/SimParticle pointers and selected-particle set.
Parse the observed membership ledger, reconcile every raw ProductID/index once as accepted or rejected and every
output constituent count/order. Exercise empty/multiple collections, duplicate tag failure, threshold equality,
noncompress mode, equal-time sort ties and both time-gap and final-bucket emission. Compare stable replay with the
newly observed order without changing stock sort or repairing associations. The pilot's strict replay failure
remains failure evidence until this independent control resolves its scope.

## E. Native recorder fixture matrix

Only after explicit runtime/site/budget approval run bounded fixtures with known topology, comparing diagnostics
off/on for unchanged stock products and RNG state where exportable. No expected physics/RNG invariance is inferred
from absence of random calls alone. At minimum cover:

| Fixture | Required observation |
|---|---|
| Outside track never reaching a root | begin/end, no entry; not a certified physical zero |
| Entry and termination inside | native pre/post transition plus terminal record, no invented exit |
| Exit/reentry | same event/root lineage, increasing visit, escaped state not an independent draw |
| Secondary born inside, including at rest | birth observation or explicit unavailable-native-touchable failure |
| Null world boundary | explicit world-null endpoint, not lost/made-up exterior volume |
| Shared/tangent/zero-length boundary | retain native evidence; no tolerance deduplication |
| Suspend/postpone/resume | explicit lifecycle and unsupported cross-event status until validated |
| Empty, rejected, aborted, overflow, killed track | independent manifest/status reconciliation, never successful empty promotion |
| Missing/duplicate root, duplicate output filename | hard setup failure |
| Row/byte cap, bad output stream, interrupt | incomplete artifact retained, no success footer synthesized |

Use a pre-response source manifest separate from recorder-surviving rows. Require exact begin/end and closed art
EventAuxiliary/StatusG4 reconciliation, original source-artifact checksum, root/parent remapping across stages and
all product registries. The boundary draft's local event markers are not themselves this external manifest.

## F. First library and hybrid validation

A successful recorder test does not establish library support. Classify initial upstream siblings versus returns
using native transition history and persistent ancestry; keep no-entry/born-inside/failure ledgers. Quantify exterior
feedback before claiming a standalone kernel. Build only the supported, reviewed joint-event representation with
explicit unsupported cases and a source-artifact split. Then inject its CSS into stock response/reconstruction and
measure assigned CE cluster energy peak/width/tail, time/position, E/p and match efficiency with paired per-shard
errors and one held-out confirmation. Cosmic MIPs follow. Full JAX digitization/reconstruction and performance
measurement remain separate work; no `cesob` promotion occurs at these recording gates.

No event count, production cost estimate or new campaign is authorized by this plan. Propose a bounded campaign
only after source/runtime/native-geometry gates establish what the existing artifacts cannot provide.
