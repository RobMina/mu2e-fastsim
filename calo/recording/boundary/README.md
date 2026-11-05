# Default-off native candidate-boundary recorder — review artifact

**Original preparation snapshot:** this directory's patch/receipt describe a paused-source review, not current
assembled-source status. The refined implementation was subsequently user-built and loaded in a separate area;
native boundary lifecycle/transition execution and containment remain unvalidated. See `../../PAUSE_RESUME.md`
and `../current_head/README.md`. No transport is approved.

For this original delivery, all files lived in this directory: no actual Offline edit, remote command, build,
transport, staging, commit, install, ROOT product or dictionary generation was performed at that stage. `../../BOUNDARY_RECORDING.md` remains the design authority; this artifact implements only a diagnostic lifecycle/transition hook, not its acceptance gates.

## Artifacts and local source-only check

- `boundary-recorder.patch`: unified patch with paths relative to an Offline checkout.
- `payload/Mu2eG4BoundaryRecorder.hh`: reviewable mirror of the added header.
- `source-manifest.json`: SHA-256 of each changed base source, read reference headers/sources, CMake/SCons manifests, expected patched files, and the patch itself. These hashes identify file bytes, **not a git revision or clean/current checkout**.
- `VALIDATION.txt`: portable source-only receipt, interpreter version and relative-file SHA-256 values, with explicit untested native gates.
- `make_patch.py`: reproducible insertion-only patch generator; writes this artifact directory only. Do not regenerate the manifest merely to waive a mismatch: review/rebase the source first.
- `check_patch.py`: read-only applicability preflight against Offline, then patch-executable dry run, actual apply, structural assertions, deliberate temporary drift rejection, reverse patch, and original-source hash recheck. All patch commands run in a disposable temporary directory containing copies, never in Offline. It also mutation-checks rejection of missing root validation, missing/duplicate-name guards, and the repeated-root guard. These are structural checks, not native G4 tests. It does not compile anything or assert native execution.

From the project root, set `PROJECT_VENV` to the project virtual environment and `OFFLINE_CHECKOUT` to the explicitly selected paused checkout, then run without building:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PROJECT_VENV/bin/python3" calo/recording/boundary/check_patch.py \
  --offline "$OFFLINE_CHECKOUT"
```

Both scripts require explicit `--offline`; there is no checkout default and no machine/site path in the manifest. The test requires Python 3 and a `patch` executable supporting `--batch`, `--fuzz=0`, and `--dry-run`. Success is a text/hash/patch applicability result only. A source mismatch, new-header collision, offset, or fuzz is a rejection, not automatic adaptation.

## Patch footprint and dependency policy

The patch changes exactly:

1. `Mu2eG4/inc/Mu2eG4Config.hh` — four optional debug defaults.
2. `Mu2eG4/inc/Mu2eG4SteppingAction.hh` — owns the header-only recorder and exposes it to sibling actions.
3. `Mu2eG4/src/Mu2eG4SteppingAction.cc` — constructs from debug config; records before stock stepping bookkeeping/cuts.
4. `Mu2eG4/src/Mu2eG4TrackingAction.cc` — birth hook after `saveSimParticleStart`; end hook after `saveSimParticleEnd`.
5. `Mu2eG4/src/Mu2eG4EventAction.cc` — independent BeginOfEvent art identity; unconditional EndOfEvent record before the accepted-event branch.
6. New `Mu2eG4/inc/Mu2eG4BoundaryRecorder.hh` — all diagnostic implementation.

No new compilation unit or library dependency is added. Existing Geant4, canvas, MCDataProducts and FHiCL dependencies are reused, plus the C++ standard library and POSIX libc (`mkdir/open/close/getpid`). Both `Mu2eG4/CMakeLists.txt` and `Mu2eG4/src/SConscript` were read and hashed. Neither needs a new source/dependency entry for this header-only design; existing header installation policy still needs user build review. POSIX-only output is intentional; no portability claim is made.

## Default-off configuration contract

Under the existing module `debug` table:

| Name | Default | Meaning |
|---|---|---|
| `boundaryRecorderPrefix` | `""` | Empty disables hooks and diagnostic I/O; nonempty explicitly opts in. |
| `boundaryRecorderRoots` | `[]` | Exact **physical-volume names**, matched against every node in the actual touchable ancestry. |
| `boundaryRecorderMaxRows` | `1000000` | Positive, cumulative whole-file row cap including markers and track lifecycle rows. |
| `boundaryRecorderMaxBytes` | `1073741824` | Positive, cumulative whole-file byte cap including newlines. |

Enabled configuration requires nonempty, unique exact names and positive caps. Nothing implicitly chooses `CalorimeterMother`. Selecting exactly that name would observe its actual subtree membership; that is a **candidate domain, not a certificate**. Before every enabled `begin`, including the first, the helper scans the actual built `G4PhysicalVolumeStore`. Each configured exact name must match exactly one store object: missing names (including typos) and duplicate names throw before event activation or the begin marker. A selected replicated or parameterised root also throws. This does not select an audited placement path: a unique store object can still occur below a repeated/replicated/parameterised ancestor, so store-object uniqueness does **not** certify touchable-path uniqueness. Root descendants may also be replicated or parameterised; output paths retain copy/replica evidence. No containment or complete placement inventory is certified.

This version rejects a multithreaded Geant4 application, even a one-worker MT application, when enabled. It supports sequential G4 only and checks the G4 thread ID on every event hook. There is no claim of MT correctness. Output ownership is per action/worker, not a global/thread-local recorder singleton.

## Native observation and endpoint conventions

- Membership is a union of configured exact root-name **touchable subtrees**, not a coordinate cut, VD plane, shell proxy, momentum sign, navigator query, or hardwired calorimeter certification. Only outside→inside and inside→outside pre/post changes emit transition rows; step status alone does not define membership.
- `step()` receives the real `G4Step` at the start of `UserSteppingAction`, before the stock Mu2e cuts mutate track termination state. Both complete pre/post states are retained. The selected crossing endpoint is **post step** for entry and exit. This is the native post-step state after that step's transport/process updates, not pre-material-entry momentum, a normal-offset state, an extrapolated center, or a separately navigated surface limit. Track status on transition rows is before stock stepping-only cuts; PostUserTracking reports the later status.
- `born_inside` is emitted from `PreUserTrackingAction` **after** stock `saveSimParticleStart`, using the track's currently available navigation touchable. G4 may not yet have supplied a valid touchable at this phase, particularly for some primary/version paths. Null is **unknown**, never outside or an invented inside. `track_start` then records membership `-1`; first observed stepping emits `birth_membership_unresolved` with its pre state. An inside pre observation opens an explicitly unresolved-origin visit, **not a synthetic birth or crossing**. Zero-step tracks remain unknown if no touchable is supplied. This patch does not fix that native lifecycle limitation by navigator relocation. Native-version birth-touchable availability must be measured before accepting a birth-complete ledger.
- `PostUserTrackingAction` emits `terminal_inside` only for `fStopAndKill` / `fKillTrackAndSecondaries` with observed inside membership. `fStopButAlive`, suspend and postpone are nonterminal `tracking_end` observations. A resumed tracking hook emits `tracking_resume` without resetting visits; its copied stock SimParticle metadata are not claimed to repair resumed tracking in the underlying source.
- Terminal membership uses the **last post-step** touchable/state when available, rather than a potentially stale current-track touchable. With zero steps it uses the track state. The row identifies that choice. Terminal ends are not exits; no fictional escaping energy/state is added. Unknown terminal membership is counted and remains unresolved.
- A post `fWorldBoundary` with null physical volume is explicitly `world_null`, membership outside, empty path. A non-world unavailable pre/post touchable emits `membership_unavailable` then throws: no silent drop, no inferred outside state. World-null post material is retained as returned by G4 and may be the last material; it is not a claim of a material outside the world.
- Internal steps, tangencies without a membership change and within-union root-to-root moves do not emit transition rows. Repeated and zero-length membership-changing steps are retained without deduplication/interpolation. Shared-surface and born-on-surface ambiguity is not resolved. Outward normals, placement transforms/solids and geometry surface IDs are **not implemented**; navigation validation remains mandatory.

## Ledger, identities and precision

Output is line-delimited diagnostic JSON, schema `mu2e-boundary-diagnostic-v1`. Sequence begins at zero for `event_begin`, increases for every row through `event_end`, and resets for each event. Visits start at one for an observed birth-inside, unresolved inside first observation, or entry; increment on each later entry, persist across exits, and are scoped by event + G4 track. `first_observed_entry` deliberately does not certify complete first-entry history when birth navigation was unavailable.

- Every row has independent **art run/subrun/event**, not a guessed G4 event number. Begin/end also carry the G4 event ID and begin has simulation stage, exact selected roots, candidate label, frame and units. Rows from a worker file are grouped by begin/end markers. Duplicate art EventIDs across reprocessing/stages require external provenance; the file is not a globally unique event catalog.
- Particle identity uses `SimParticleHelper::particlePtr`, never `trackID` as a persistent key. Its numeric `art::ProductID::value()` (uint32 encoding, labelled `art_ProductID_value_uint32` in begin) and numeric key are file-context references, not a globally resolvable source artifact ID. The tracking hook looks up the helper-derived key in the stock transient map after save. `sim_saved_at_start` explicitly distinguishes a pointer target actually saved from an intended key lost to stock SimParticle overflow. No pointer is dereferenced through art product lookup during transport.
- Start/born rows carry the **saved** SimParticle parent Ptr and GenParticle Ptr when present, alongside G4 local track/parent IDs, PDG and direct creator process name. This handles multistage persistent parents without fabricating a track-zero parent. `null` on secondaries' GenParticle links is expected; generated-root identity must be reconstructed by joining recorded persistent ancestry and independently archived generator/prior-stage products. No designated-root traversal/causal assignment is implemented.
- Each start/born row records native position, momentum, kinetic/total energy and global/proper/local time. Each crossing retains both pre/post states, step statuses, step length, track status, direct post process name, material name/index, and root→leaf PV/LV paths with both copy and touchable replica numbers. Lifecycle track-state material is labelled as touchable logical material, not falsely advertised as a step material (important for parameterized volumes). Native step-point material is used on transitions and post-step terminal rows.
- Numeric state uses `double` and `max_digits10`, native G4 world coordinates (not Mu2e-origin-subtracted), `units="mm_MeV_ns"` conventions, matching the Python recording schema. Integer enums are native G4 version-dependent statuses. There is no downcast to float; nonfinite state/step-length values throw rather than emitting invalid JSON numeric tokens. Geometry material indices and paths are meaningful only with the external geometry/configuration provenance. Names are JSON byte-escaped when non-ASCII/control bytes appear.
- Begin is written before detector-product creation/cut begin-event hooks. End is emitted for completed G4 EndOfEvent callbacks regardless of response acceptance, including **zero tracks or zero transitions**, after the existing filter decision and outside its accepted branch. End contains aborted flag, stock-filter decision, G4-equivalent status code (overflow 10, killed-step/field 1, otherwise 0), counts, unclosed visits, unknown birth/end memberships, and missing saved particles. This is not an actual persisted `StatusG4` product or proof of successful later art product insertion. The stock status/counters and filter decisions are not changed.

The helper never calls `PhysicsProcessInfo::findAndCount`. Direct G4 process names are observational reads. Existing stock calls to process counters remain untouched. It does not alter momentum, track status, geometry, cuts, RNG, trajectory controls or normal art products. This is an intended source-level invariant, **not measured physics equivalence**; enabling I/O changes timing and diagnostic failures abort execution.

## Exclusive output, bounds and incomplete events

For an enabled sequential action, create `<prefix>.pid<PID>.worker<G4ThreadID>/` with exclusive `mkdir`, private mode 0700; inside it exclusively create `boundary.jsonl` with `O_CREAT|O_EXCL`, mode 0600. Any collision/permission error throws. Reopen that empty file via `std::ofstream` **append**, never truncation, inside the newly owned directory. The private-directory scheme avoids an ordinary writer's create/reopen race, but is not a hostile same-UID filesystem defense. Use a trusted local directory; file ownership, retention, disk quotas, and collection remain the operator's responsibility. There is no overwrite fallback or automatic resume/merge. Multiple enabled modules with the same prefix/PID/worker collide deliberately.

Every emitted row is flushed and stream state checked. Whole-file bounds are checked before each row and throw on overflow; there is no truncation, thinning, limit-driven transport cut, or partial-row success claim. Construction of a row and track bookkeeping are not separately memory-bounded beyond rows/normal transport limits. The file cap includes lifecycle rows for all tracked particles, even outside the domain. Failures are intentionally hard, not swallowed.

If a cap, exception, disk error, process termination or failure before EndOfEvent prevents the end hook, the event has no trustworthy `event_end`; the next begin also rejects an unclosed event. Flush is not fsync/crash durability. There is no destructor-synthesized success marker and no promise to manufacture an end marker after fatal failure. Consumers must reject missing/mismatched markers and keep incomplete/aborted events in a failure ledger. A completed end row with unknown memberships, overflow, killed tracks, or unclosed visits does not make a complete containment ledger.

## Original acceptance gates — before any transport

This list is the original review plan. Current completion is in `../../PAUSE_RESUME.md`: source rebase, user build,
structural inventory and bounded generator-only manifests have since progressed; containment and native boundary
transitions have not. Do not treat this list as instructions to repeat completed work.

1. Review/rebase against the actual intended source, then separately authorize a user build. Validate FHiCL unsigned-long-long types, canvas ProductID encoding, all native G4 APIs and lifecycle/order assumptions on the intended versions; no C++ compilation was performed here. Confirm source header installation in both build systems. Validate stream failure/collision/bounds behavior with native tests.
2. **Native containment/placement inventory:** export actual placed solids, transforms, parent/daughter placement identities (including replicas/parameterizations), material assignments and geometry configuration hashes. Audit CalorimeterMother's descendants against the intended two-disk passive/FEB/cable domain. Explicitly include or assign **remote calorimeter cable volumes**, not just the local cable objects in `constructDiskCalorimeter`; trace `constructDS`/services and other cable placements outside that mother. Verify radial/phi surfaces, gaps, shared surfaces and all proposed included volumes. A name match, GDML picture or shell union is not containment evidence.
3. Review **cechain ownership** of upstream/material transport, remote cables and shower branching versus the calorimeter response domain. State where upstream transport ends, which material belongs to whom and how crossing/reentry lineages retain the same independent event. Avoid double transport, missing material and silent reassignment. This recorder supplies no ownership decision.
4. Native diagnostic tests must cover known entrance, exit/reentry, inside birth, unavailable birth touchable, born-on-surface, world exit, zero-step termination, stop-but-alive, suspend/postpone, zero-track/zero-transition events, aborts, overflow and filter rejection. Confirm post-step endpoint semantics against same-step stock response and that files preserve actual navigation history. Suspended/multistage histories and cross-event postpone linkage need explicit support review.
5. Bind files to immutable input-artifact checksums, source pool/subset, source EventAuxiliary registry, geometry/material/config hashes, binary/tool versions and module/product provenance. Reconcile all input events including callbacks never reached and later framework failures. This boundary patch alone supplies no such external writer or validator; separate manifest writers and structural
audits now exist, but full independent source/status and persistent causal-root reconciliation remain open.
6. Only after those gates consider a separately authorized bounded native run and enabled-vs-disabled same-seed physics/RNG/stock-output controls. Separate raw-to-compressed membership instrumentation and diagnostic entrance-set containers now exist; native
membership correctness/invariance, sufficient correlated conditioning, persistent generated-root assignment,
causal lineage ownership, normals/surface certification and response-library validation remain open.

No placement inventory or native containment closure is claimed by this diagnostic patch.
