# Target source/POT adapter and callback review

Source-level proposal only: **A/B and in-memory reconciliation have an unapplied, uncompiled v2 draft; a separate uninstalled original-product reader and structural join are now proposed, but actual written-product verification, causal frontier, native qualification and POT certification are not delivered**. The default-off [native draft](native/README.md) remains unapplied, uncompiled and library-ineligible. This document narrows the work needed for [target-library acceptance](TARGET_LIBRARY.md); it does not authorize a build, geometry construction, native probe or Geant4.

## Reviewed source boundary

Offline source is pinned to `b24d4163e5e8f5ed2b5f72fccb25e4980fec64e2`. Geant4 implementation review uses upstream `v11.3.2`, annotated tag `1b1ba3154688256fe92c44fecc455050f00f64c9`, commit `62f62ecae238a7c304c52af4affbe70795475590`. The tag is unsigned. Source hashes and paths are in `native/semantic_source_pins.json`; these are additional review evidence, not extra applied patch files. Published-version source is **not** proof of exact installed manager implementation bytes. Installed `G4Event.hh` was separately checked: `IsAborted()` exists with matching header bytes. No manager implementation or native callback was executed.

### Callback conclusions and limits

| Source observation | Consequence for this proposal |
|---|---|
| `G4EventManager::DoProcessing` calls stack preparation before event begin, then stacks transformed primaries. | Ordinary primaries are classified after recorder begin. Postponed histories can be classified earlier; they remain unsupported and must not be adopted into a new event. |
| `StackTracks` assigns secondary IDs before `PushOneTrack`, which invokes classification. | Birth-token resolution at classification is supported by upstream source order, not by a species/order/time association. Transient pointer lifetime still needs a native control. |
| `G4TrackingManager::ProcessOneTrack` calls post-tracking before returning to the event manager; `fStopAndKill` then stacks children. | Parent end precedes child classification. A production step can be linked to a later assigned ID without assuming callback order equals physical birth order. |
| `fKillTrackAndSecondaries` causes the event manager to delete the secondary bucket without stacking it. | Refuse that terminal state when observed offspring remain unassigned. Do not erase tokens as if classification occurred; a future disposition extension needs explicit accounting and tests. |
| `G4RunManager::AbortEvent` and hard `AbortRun` mark the event aborted and clear the stack through the event manager. Direct event-manager abort clears stacks without itself setting `G4Event`'s flag. | Observe `IsAborted()` and refuse true, but false is not an independent all-history certificate. Reconcile primaries and tracked/stacking outcomes. |
| `GetSecondaryInCurrentStep` exposes the surviving secondary-bucket suffix; zero-KE children without an at-rest process can be deleted before entering that bucket. | The draft observes transport-eligible bucket births, not every microscopic production-process secondary. Do not claim exhaustive process-level birth coverage. |
| Sensitive-detector hits precede the global stepping callback; regional stepping follows it. | The draft is before Mu2e common stepping cuts, not before all possible callbacks or status changes. Record end state separately; qualify any regional/custom tracking/fast-simulation path before admission. |
| Transportation updates post touchable/material only while positive kinetic energy remains; post-step processes can subsequently alter state. | Native pre/post observations are retained, not promoted to an exact production vertex or universal next-volume claim. Same-step boundary production stays unresolved. |

The source draft now carries an explicit native abort input and moves its end hook after SD handoff. Its marker scope is recorder end-hook plus flushed/closed ledger; `putDataIntoEvent`, `TerminateOneEvent`, framework callback return and file output are later gates. A partial directory is never a zero event.

## Adapter boundaries and minimum rows

The v2 diagnostic draft implements proposed hooks for A, B and the assembled in-memory collection. Preparation reserves the bounded per-EventID ledger before `BeamOnDoOneArtEvent`; event begin must match it. The strict auditor checks the source-row schema, exact populations and identity links without promoting them to native or persisted evidence. The [row schema](TARGET_SOURCE_SCHEMA.md) is separate from the [original-product readback proposal](TARGET_READBACK.md), which has not read an actual art file. V1 streams remain source-unverified diagnostics and cannot acquire source reconciliation by adding a completion flag.

### A. Actual generated input before transport

At `Mu2eG4::produce` after event storage initialization and before `BeamOnDoOneArtEvent`, read the **configured** `GenParticleCollection` handle using `ioconf.inputs().primaryTag()`. Capture EventID, actual ProductID and branch/process identity, every collection key and persisted PDG/generator ID/position/four-momentum/global and proper time. Include collection cardinality, explicit empty collection and actual proton count. Missing handle or invalid provenance prevents reconciliation; never infer one proton from an EventID or request count.

`GenParticle` has no weight field in the pinned definition. Keep generator/event statistical-weight products, primary conversion weights and native track weights distinct. Unknown weighting provenance stays unknown. The current primary converter passes the three-momentum into a mass-shell G4 primary; although its helper signature accepts proper time, the reviewed body does not assign that argument. Capture persisted generator state and actual converted G4 state independently rather than asserting all fields transfer unchanged. The `testPDGIdToGenerate` path can add an extra primary without a helper entry; initial supported production configuration must exclude it explicitly.

### B. Authoritative primary and SimParticle outcomes

`Mu2eG4PrimaryGeneratorAction` adds an actual `art::Ptr<GenParticle>` to `SimParticlePrimaryHelper` for each ordinary generator entry. `getEntry(g4TrkID)` indexes its table at `g4TrkID-1`; use the existing primary-helper association, not an independently reconstructed index rule. At primary classification, record the actual generator pointer ProductID/key plus assigned native ID and classification, including killed roots that never get a SimParticle. Validate handle ownership/availability and conversion multiplicity against receipt A.

At the actual `saveSimParticleStart` insertion point, record inserted versus size-limit omission, collection ProductID/key/stage, native ID, actual `parentPtr`, `genPtr` and creation code. A hook after this function is **not** insertion evidence: its overflow branch returns early. `SimParticleHelper` coordinates are expected addresses only; staged inputs may add a key offset. Keep the initial adapter restricted to GenParticles, with staged/remapped inputs refused.

After `_trackingAction->endEvent()` assembles `simPartCollection`, join against the actual collection entries and compare parents, generator links, stage, key and reciprocal daughters. Distinguish inserted particles, native stacking kills, unsupported child disposal, omission and unresolved links. This is in-memory evidence only. Do not dereference an output pointer before its product exists and call it persistent validation.

### C. Written products and whole-shard closure

A separate readback of the **original written products** must bind file content/registry/process provenance and reproduce A/B identities, all source rows, native outcomes, parent/daughter links and zero-event presence. It must check configured output selection/drop rules, not assume a raw-output label is inclusive. Bind resolved source/runtime/config/geometry receipts and complete termination logs. Missing output, write failure, abort, overflow or ambiguous link makes the intended shard incomplete; do not normalize surviving events as an unbiased subset.

`GenEventCounter::produce` increments once per module callback, not per proton or completed G4 event. Its end-SubRun product is `GenEventCount`, it refuses conflicting pre-existing count products, writes then resets its counter. Reconcile distinct SubRun products and event sets; never sum copied parent counters or overlapping shards.

For an explicitly verified one-proton/event, unit-weight, unfiltered configuration the required set closure is:

`requested EventIDs = generated-source EventIDs = begun native EventIDs = complete native EventIDs = written/read-back EventIDs`.

For each event, every actual generator entry has one verified conversion/classification outcome, and every transport-accepted root has a complete supported native history. Proton counts, not crossing counts, define generated POT. Report generated and completed proton counts separately until all sets close. A general variable-multiplicity or weighted source needs its own reviewed normalization policy; empty generator events are not automatically one POT. Development and held-out shards must be disjoint by source identity, not just output filename.

## Causal frontier derived only after reconciliation

Use event-local continuation-step edges and observed daughter-production incidence, not callback row/time sorting. A candidate outward union transfer owns the parent's subsequent continuation and births strictly after it. A daughter born on an earlier parent step remains a separate active branch. Same-parent-step boundary birth is **unresolved** without process/production-vertex ordering; no timestamp, momentum or endpoint interpolation resolves it. Persist candidate/admitted/suppressed/unresolved with owner/reason. Keep diagnostic rows even when suppressed.

The target-only frontier cannot silently own earlier exterior branches or target-missing primaries. An explicit upstream owner is needed, and the transport arm must retain target material dynamics for return/feedback. Stored stock returns and post-transfer descendants cannot be replayed as extra launches when exterior transport changes. No frontier output exists yet.

## Approval and next implementation gates

1. Independently source-review and complete the unapplied A/B and in-memory v2 draft and the separate original-product reader/structural join, including their strict schemas and source pins. Actual-art verification, authenticated provenance, output-selection and whole-shard exposure reconciliation remain undelivered. Do not turn structural consistency into validity flags.
2. Qualify inventory selectors and active-material union separately; no inventory has been constructed here.
3. Prepare a complete isolated source area only when the draft is reviewed/build-ready; the user performs the build.
4. After build freshness checks, request distinct native/inventory and one-event-first transport controls with explicit event/site/CPU/wall/output/recovery caps. Existing 100-event FCL is not ready for this recorder.
5. Only successful persisted controls justify a separately sized inclusive production request. No production library, PS/TS material closure, fast rate or physical event-cost measurement follows from this source review.
