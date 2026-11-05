# Assembled current-head source contract

**Current checkpoint: R6 paused 2026-11-05.** The compact implementation is user-built and its zero-transport
structural inventory passes. Begin with `../../PAUSE_RESUME.md`; earlier build failures/rebuild requests below are
chronological evidence, not new instructions. Hash-pinned preparation manifests retain their original fields.

The prepared Offline source base is canonical revision `b24d4163e5e8f5ed2b5f72fccb25e4980fec64e2`, with its own
recommended envset **p110**. Do not run these modifications against the older p109 saved-step control area.
The separate preparation leaves all modifications unstaged; no commit, build, or transport is included.

`source-manifest.json` pins the current base and final assembled file hashes. The base boundary/membership patches
are byte-identical to their earlier review patches because their edited sections did not change. Configuration
changes elsewhere, the new event-seeding build source and canonical build metadata are preserved explicitly.
This is an intentional source review, not a general hash-waiver mechanism.

The boundary helper also receives `lifecycle-refinement.patch`: initial track start requires step zero, stepping
requires a running nonterminal track, unresolved birth is first-step-only, resume requires prior suspension and
matching membership, and terminal/running state is retained explicitly. The installed p110 Geant4 11.3.2 enum has
`fSuspend=4`, `fSuspendAndWait=5`, **`fPostponeToNextEvent=6`**. Python ledger validation requires the explicit
`geant4-11.3.2-p110-G4TrackStatus-v1` provenance convention; native C++ uses symbolic enum names. Cross-event
postponement is not made supported by this work.

`prepare_current_head.py` (one directory up) prepares only clean exact-base clones, checks reviewed drift and
applies zero-fuzz patches without building. `refine_lifecycle.py` refines only its hash-pinned helper.
`assemble_manifest.py` copies the two manifest writers into a distinct `CaloRecording` Offline package, adds
that package to top-level CMake, and checks the inventory integration. Its CMake and SConscript directly list
matching art/TFileService dependencies, including art Utilities. The inventory helper is header-only in Mu2eG4;
existing CMake header installation and SCons source discovery cover it without new link dependencies.

Verify the prepared area with an explicit checkout path:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 calo/recording/current_head/verify_area.py --offline /path/to/isolated/Offline
```

This checks source hashes, exact base/checkpoint tree, index/deletion state and whitespace—not compilation.
At the original base, intended changes may be unstaged. A committed checkpoint is also accepted only when the
base is its ancestor, its complete changed-path set and every committed patched blob match the unchanged source
manifest exactly, and its working tree/index have no tracked or unexpected untracked changes. No broad descendant
or branch-name waiver is used. The pause checkpoint changes Git metadata, not the built native source or pins. Setup helpers
contain no setup/build command. Builds belong to the user on the designated build host; runtime/configuration,
geometry-only inventory and same-input disabled/enabled tests follow only after confirmed build completion.
The review FHiCL still contains deliberately invalid hash tokens; no runnable transport campaign is supplied.

## Partial build against the exact published CI

`prepare_overlay.py` creates a separate build-only area containing **CaloMC, CaloRecording and Mu2eG4**, backed by
published CI `main/b24d4163_94a4d81f_5999ffa2` (`al9-prof-e29-p110`). The full prepared source checkout remains
intact and unstaged. Overlay directories are real, while every source file links to the prepared checkout; this
is necessary because Muse discovers SConscripts without following directory links. Edit the full source checkout,
not an independent copy. Reprepare a new overlay if files are added or removed; verification rejects inventory drift.

```sh
python3 calo/recording/current_head/prepare_overlay.py \
  --offline /path/to/prepared/Offline \
  --backing /path/to/exact/published/ci \
  --overlay /path/to/new/build_overlay
python3 calo/recording/current_head/prepare_overlay.py \
  --overlay /path/to/new/build_overlay --verify-only
```

The preparer requires the exact backing Offline revision and matching environment, a completed build receipt and
required libraries/executable. Every ordinary selected-package file must match that backing unless it is a pinned
prepared override. It creates a backing symlink and per-file SHA receipt; it does not run setup or compilation.
Only three package SConscripts are exposed, so the normal user build invocation builds those local packages, not
all Offline. Other headers, libraries, data, configuration and the executable come from the exact backing.

Run the user build from the **overlay work directory**, never its full-source parent. In a fresh shell:

```sh
cd /path/to/new/build_overlay
# Initialize the experiment environment and Muse using the site's standard commands.
muse setup
muse build -j 32
```

Setup alone was checked on the native host: p110, local library/plugin paths preceding the exact CI paths, and
backing executable resolution without executing it. Muse emits its generic repo-order warning because the backing
also contains Production and mu2e-trig-config, giving `Offline Production mu2e-trig-config Offline`. These two
backing repos provide configuration/scripts, not replacement compiled Offline code; the selected Offline base is
identical. The warning is recorded, not suppressed or treated as a compile/link certificate. The three modified
packages must still be built together and actual library/plugin resolution verified after user completion.

## First user compile correction

The first user build reported `-Werror=sign-compare` in the inventory daughter traversal. Installed p110
`G4LogicalVolume::GetNoDaughters()` returns `std::size_t`, and `GetDaughter` takes that same type. The corrected
header uses a `std::size_t` index and explicitly includes `<cstddef>`; no cast or warning suppression is used.
`fix_inventory_index.py` applies only those two hash-guarded edits, checks the corrected review payload, preserves
the prior source/overlay receipts, and refreshes their pins. It never compiles. The overlay source-file link sees
the corrected full-checkout header automatically. Structural regressions require the matching index type/include;
a successful user rebuild and subsequent native checks remain pending.

## Manifest plugin direct-link correction

The next user build reached the manifest plugin link stage and reported an undefined reference to
`cet::detail::provide_file_path[abi:cxx11](char const*)` in both modules. Native p110 symbol inspection confirms
`libcetlib.so` exports it; art's module macro invokes the file-path helper. Both build descriptions now directly
list `cetlib` / `cetlib::cetlib`, in addition to the distinct `cetlib_except` library. SCons does not infer this
missing direct dependency transitively. `fix_manifest_linkage.py` applies only the two reviewed library-list
additions, preserves the prior receipts and updates source/overlay/review pins. No module behavior is changed.
Source tests require the direct dependency in both files. The user subsequently reported a successful build;
post-build checks are summarized below.

## User-completed build verification

The user completed the exact-CI-backed p110 build after both corrections. `verify_build.py` records the completed
build receipt, source/overlay hashes, seven library/plugin SHA256 values, binary freshness against package inputs,
local-before-backing search paths, direct manifest cetlib NEEDED entries and resolved dependency paths. Both
Mu2eG4 plugins resolve the modified Mu2eG4 main library locally, and the membership producer resolves local CaloMC.
Both manifest plugins and the CaloMC targets have clean isolated relocation checks; no checked library is missing.

**Mu2eG4 does not have a clean isolated `ldd -r` result:** all three checked Mu2eG4 targets expose 108 GSL `cblas_`
symbols, identical to their untouched exact-CI counterparts. The verifier retains the actual symbol lists and
fails on any new/different symbol or missing dependency, rather than relying on ldd's exit status. The backing
executable itself has clean isolated relocation checks. This stock-identical caveat does not prove the final art
global-loading context works and does not authorize a preload, dependency edit or transport run. Verify that
context in a separately gated native configuration check. No physics, FHiCL, ROOT persistence, geometry or normal-
product invariance result follows from successful compilation/linkage.

Preparation manifests retain their preparation-time `build_performed=false` and rebuild-pending fields; the
separate post-build receipt records the subsequent user-reported success and measured binary checks. Native
execution and physics-validation fields in that receipt remain false.

## First no-transport native checks and required corrections

Framework `--print-description` loaded all four new/modified modules from this overlay. A one-art-event job with
an always-false FixedFilter before sequential Mu2eG4 reached native run initialization with no transported event;
its inventory failed closed with `serialization byte cap`, zero framework events processed and an empty boundary
ledger. No inventory file was created. This message did not distinguish cumulative file budget from the separate
64 KiB native StreamInfo budget. Corrected source adds bounded stage/buffer/remaining-byte/placement context to
failures without increasing any cap. Full-world buffered inventory still has substantial memory overhead beyond
serialized size; no solid/containment certificate exists.

The first generator-only manifest fixture was rejected during art configuration: **EDAnalyzers are forbidden on
trigger paths**. Both writers are now reviewed as **always-pass EDFilters**, returning true only after attempted
rows and existing missing/multiplicity checks, with FHiCL declarations moved to `physics.filters`. Labels, schema,
float64 states, consumes and row order are preserved; plugin kind and framework filter bookkeeping change, so no
framework-output identity is claimed. End-path relocation would violate the pre-response denominator contract.
`fix_native_integration.py` preserves previous receipts, archives the exact corrective patch, checks hashes and
updates source/overlay/review pins. The user must rebuild before further runtime checks. The previous successful
build receipt is historical and intentionally rejected by `runtime_checks.py` after source changes.

`runtime_checks.py`, `zero_transport.fcl` and `manifest_fixture.fcl` supply bounded checks and preserve failures in
fresh output directories. No G4 event is requested: geometry uses one rejected art event with empty macros; the
manifest fixture reads a hash-pinned stopped-muon pool with a generator but no resampler or G4 module. Initial FCL
syntax/reference failures were corrected and retained, not counted as passes. No successful manifest ROOT output,
empty/missing/multiple/generator-exception integration or membership off/on invariance result exists yet.

## Corrected user build and bounded native results

The user completed the incremental rebuild after the filter and cap-diagnostic corrections. A fresh verification
receipt binds the unchanged corrected source/overlay pins to seven fresh local binaries, the completed build,
local plugin/main-library resolution and exact backing. The isolated stock-identical CBLAS caveat is unchanged.
All four module-description checks pass; both manifest writers identify their plugin kind as filter.

The zero-transport check again reaches native BeginOfRunAction with zero processed events, then fails at placement
163,069: committed bytes 268,434,777 of 268,435,456, remaining row buffer 679 bytes, used 672, attempted write 9.
This diagnoses the cumulative **256 MiB file budget**, not the separate StreamInfo limit. No inventory is created;
the boundary ledger is empty. Observed high-water memory is about 2,032 MB, not bounded by serialized file size.
No caps were raised, compiled sources changed, geometry certified or G4 event processed.

Generator-only single-root, strict-single and valid-empty jobs each close successfully for three independently
expected source EventIDs. External ROOT audits pass event order, registry/root identity, native double state and
metadata/source/normalized-FCL claims. Strict-empty, missing prior-process product and generator-error fixtures
reach their intended exceptions, with observed prefixes of 1/1/0, 1/1/0 and 1/0/0 input/incident/root rows. These
prefixes remain incomplete diagnostics, not crash-safe completion evidence. An earlier missing-current-producer
fixture fails framework configuration before callbacks and is retained separately; the prior-process variant
exercises the actual missing-product callback without changing C++.

`audit_fixture.py` (one directory up) writes exclusive structural-audit receipts bound to actual ROOT/log/config/
source/runtime/build hashes. It never runs art or transport. Multiple-root and resampler-rejection runtime cases,
independent generator-state replay, raw membership off/on and native containment/transition gates remain open.

## Compact v2 inventory — renewed user build gate

`fix_inventory_compact.py` applies a hash-guarded replacement of only the installed inventory header, archives the
previous header/source/overlay receipts and exact delta patch, and refreshes pins. The optional hook, dependencies,
three-package overlay and all manifest/response/boundary/membership sources remain unchanged. The prior verified
build and native manifest results remain historical evidence; runtime checks reject that receipt for current sources.

Compact schema v2 uses expanded preorder placement IDs and parent references, plus material/solid/LV dictionaries
interned by native pointer encounter (pointers never serialized). All world placements and unsupported metadata
remain, with unique root store/path and sibling-name/copy checks. Native state is captured once into a compact
PMR snapshot; exact serialization preflight then output both use that snapshot, with no second native walk. A
min(file-cap, 256 MiB) helper requested-live-allocation guard includes transient growth. This is not a process RSS
bound. All existing configured file/row and hard geometry/text caps remain unchanged.

The user subsequently completed the incremental build from this same overlay. Fresh verification passes current
source/overlay hashes, all seven binary freshness checks, direct cetlib dependencies and local modified-mainlib
resolution; stock-identical isolated CBLAS caveat remains. All four module descriptions load. The zero-transport
fixture closes with one rejected art event, zero G4 events/ProcessOneEvent work and an empty boundary ledger.
The independent v2 reader accepts the complete structural placement inventory: 167,623 placements, 80 materials,
3,910 solids, 4,382 logical volumes, 175,997 rows and 49,035,617 bytes. Each of 13 roots appears once and 14,623
placements are union-selected. There are 1,143 unsupported-solid placement occurrences; CalorimeterMother remains
an unresolved G4UnionSolid. Material composition, containment/navigation and membership invariance are not certified.

Observed VmHWM is 1,695.52 decimal MB and VmPeak 1,868.78 MB. The earlier failed v1 attempt reached 2,032 MB VmHWM;
these are separate runs, not an equal-work benchmark or general memory bound. All configured caps remain unchanged.
`../audit_inventory.py` requires the matching successful runtime/build/config receipts, one rejected art event,
zero G4 processing and empty boundary before supplying exact-SHA completion evidence to the independent reader.
Checked fsync/close throws plus successful native job provide closure evidence, not a footer alone. Its exclusive
audit output binds source/binaries/config/geometry-entry-file hashes, logged expanded native geometry hash and
counts; authenticity and complete include/material provenance remain separate. See `inventory/README.md` for the
versioned format and independent structural-reader limitations. Preparation-time rebuild fields in pinned source
manifests are historical; the separate post-build receipt supplies current completion evidence.

## Remaining native validation gates

- Compact v2 compiled/loaded and completed a structurally audited zero-transport inventory; bounded manifest
  configurations executed on the preceding build with byte-identical manifest binaries. Native solid/navigation
  validity, boundary transitions, compression membership and normal-product invariance still need validation.
- Native inventory uses an all-world tree walk; any replica/parameterization/non-placement anywhere fails closed.
  Unknown solids are metadata-only. `CalorimeterMother` is a boolean union and remains unresolved solid topology
  until operands/transforms receive a separately reviewed exporter or native navigation check. A complete path
  tree is not a full solid/material-composition inventory or containment certificate.
- Input-manifest hashes identify configured claims. Independent launch/source sequence/closed-output receipts
  remain mandatory. ROOT buffering after exceptions is not a complete manifest; the external validator requires
  full independent expected EventIDs and matching incident/product/root sequences, never an observed prefix.
- StageParticle creation states are in **Mu2e** coordinates; boundary states use **G4_world**. The future adapter
  must apply the audited world-origin transform. No training data are produced by merely aligning event IDs.
- Source pool entry choice, persistent generated-root/SimParticle reseating, exterior feedback, initial-sibling
  versus returning-descendant classification, physical-zero certification and independent raw-membership control
  remain separate gates. No library eligibility or default promotion follows from a successful build.
