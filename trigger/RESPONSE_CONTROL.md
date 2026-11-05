# Stock-step response control for the CPR discrepancy

The completed development hybrid changes both the tracker steps and their response RNG history. It cannot
separate those effects from its retained original G4 calorimeter response. Its representative CPR CE path
passes 14,813 stock versus 13,185 fast events out of 28,044; 14,964 versus 13,474 reach the final track filter.
Changing final thresholds is not justified by this discrepancy or by the saved last-filter endpoints.

## Reuse two completed arms; separately declared additional control

| Arm | Tracker input and response | Purpose |
|---|---|---|
| S (saved) | Full stock steps, original production digitization | Existing reference; do not rerun |
| F (saved) | Primary-only fast steps, separately seeded released stock digitizer | Existing development hybrid; do not rerun |
| C (first parent completed; remainder paused) | Full original exported stock SGS through the same existing importer and separately seeded digitizer as F | Bound the combined response/importer difference without another transport run |

C is **imported full-stock steps with separately seeded stock digitization**, not an electronics-only
control on the present evidence. The user authorized proceeding with this control after reviewing the design. No new Geant4,
owner-source change, build, threshold tune or correction is included. Execution requires separate immutable
preparations/outputs and the complete admission below. A prefix can test runtime compatibility, not certify
full-history decisions. Existing completed S and F are reused, never rerun as part of C.

### Current execution boundary — 2026-11-05 06:10 EST

The user has paused R7; no R7 jobs are running. The first C parent (ordinal 1 of ten)
completed with exit 0 in about 10 min 32 s, processing 2,811 events and 138,045 SGS
rows, including 11 zero-SGS events. Ordered SGS numerics and generator data round-trip
exactly; all six original products, 56 original status words and retained original SGS
serialized bytes are preserved. This establishes bounded execution/preservation, not
physics validation, lossless imported particles or electronics-only isolation. The
remaining nine parents and complete-history C trigger have **not launched**. Explicit
user authorization and the [resume gates](PAUSE_RESUME.md) are required to continue;
the admission procedure below is not permission to launch during the pause.

## Importer contract and limitations

The reviewed existing `StrawGasStepsFromTable` reads every particle and step row, not only primary electrons.
With key offset zero, numeric particle/parent keys and SGS particle references remain unchanged. Steps are
inserted in table encounter order; SGS positions, momentum, shape/ionization, time, energy, length and width
are copied without coordinate/time rebasing. The explicit particle-state columns are used when present.
Original SGS positions must not be shifted merely to match a conflicting README frame label.

However, imported particle pointers target a new SimParticle product. Unknown parents become null; daughter
links are rebuilt. Creation/stopping codes and several constructor fields are not restored; particle energy
is recomputed with a restricted PDG-mass switch. An event absent from the table can look like empty response
unless an independent complete-event audit rejects it. Identical numeric keys do not establish original art
product identity. These facts prevent assuming a lossless particle round trip from numerical SGS agreement.

The released digitizer source is Offline v13_41_01 `TrackerMC/src/StrawDigisFromStrawGasSteps_module.cc`,
SHA256 `76beaea3caed31852442622d051f627d755a567d8297b984bb690336528c902b`.
Its response path reads SGS coordinates/type/energy/time, conditions and EWM/PBT; SimParticle reads also
appear in MC diagnostics. This bounded source inspection is not a complete transitive dependency audit or
proof that reconstructed particle-state changes cannot affect all response/output behavior. Establish that
independently before relabelling C−S as electronics-only.

## Required admission before execution

1. Bind the existing original source-table files to their completed exact size/SHA receipts. Reconcile every
   source event with the original generated-event table and saved decision-art identities, including zero-row
   events. The existing SGS dumper does not persist collection presence and treats an absent collection as zero
   rows; independently establish source-product presence before interpreting zeros as present-empty SGS.
   Do not use fast handoff membership as the stock denominator.
2. Preserve every species, SGS encounter order, repeated straw rows and parent/key association. Check all
   SGS particle references, type/range/finite payloads, per-event declared step counts and particle-parent
   closure. Repeated rows are not automatically separately identified physical returns.
3. Use the matching original full-stock event context for each parent, retaining original straw/calo/timing
   and TriggerResults products exactly. Do not concatenate files expecting a same-event join.
4. Reuse the actual built importer/runtime bytes from F, not an owner checkout or newly built current head.
   Keep `R7FastSteps`, `R7FastSD`, process `R7TrackerInject`, seed/engine, construction/schedule, conditions
   and the declared parent-by-parent RNG resets unchanged. C has its own output files and receipts.
5. Require numerical SGS round-trip equality, finished entries/identities/order and unchanged original six
   products/56 statuses. Verify actual consumed SGS provenance. Source files and Python tests alone do not
   establish executable compatibility or a correct imported particle product.
6. Feed all ten C outputs, in original numeric parent order, through **one** all-52 trigger invocation. Advance
   its own lumi/phi counters continuously; never copy S/F counters or restart trigger per parent.

The existing primary-only `step_table_audit.py` and `injection_campaign.py` gates intentionally require key-1
primary electron steps and fast transport receipts. Do not weaken or fabricate those gates to admit C. A new
explicit full-stock table receipt/orchestration path now supplies that separate admission, as described below;
it does not weaken or bypass the primary-only F gates.

## Interpretation

- C−S combines separate response RNG/reset history with any remaining importer/state/runtime differences.
- F−C compares primary-only fast steps to full stock steps under nominally matching response machinery. It
  is not a pure secondary-omission effect: primary trajectories and other step/particle quantities differ.
- Matching initial seeds do not match individual random draws when different step populations consume them.
  Common-seed parent resets do not provide independent replicas or an independent-parent SEM.
- Report every path, both migration directions and unconditional stage reach over all stored events. Any
  original-truth population census is labelled a diagnostic, never a deployable response/correction model.

## Saved original-step census

`stock_step_census.py` joins original SGS to particles by exact event identity and original particle key,
not a time/distance or numeric-key offset. Every generated-event identity is reconciled with the completed
saved hybrid decisions; table size/SHA and all-52 state/migration admission are rechecked. This is explicitly
a truth-population diagnostic, not a deployable response or correction model.

The 28,044-event development census contains 1,376,129 SGS rows: 1,121,225 belong to the MC-primary electron
(`pdg==11 && ismcprim==1`), and 254,904 to other particles. Here `isprim` alone selects zero electron steps:
the MC primary is not necessarily the root of the saved simulation genealogy. Other-particle SGS occur in
25,390 events. There are 143 zero-row events; source collection presence is not persisted by the dumper, so
these are not independently authenticated present-empty collections. No SGS particle reference or particle
parent is dangling. Repeated `(event,particle,StrawId)` SGS rows are absent in these grouped products; this
does not prove absence of physical returns or multiple constituent passages.

For the CPR CE path, fixed original non-primary-SGS-count diagnostic strata show:

| Other-particle SGS rows/event | Events | Stock-only passes | Fast-only passes |
|---|---:|---:|---:|
| 0 | 2,654 | 180 | 99 |
| 1–9 | 19,535 | 1,591 | 299 |
| 10+ | 5,855 | 371 | 116 |

Thus the deficit is not restricted to events containing other-particle SGS. These truth-defined subsets are
not randomized controls or independent replicas: primary transport/returns, response RNG and retained stock
calorimeter history still differ. The result does not quantify a secondary contribution, justify a scale or
threshold adjustment, or replace C. All stored events and both migration directions remain in the report.

A useful control can bound mechanisms without proving a cause. Preserve uncertainty instead of tuning cuts,
renormalizing efficiency, removing difficult events, or claiming independent confirmation on development data.

## Separate admission and numerical round trip

`response_table.py` is a separate full-stock admission, not an option to the primary-only audit. It validates
all original generator/particle/SGS identities, encounter order, all species, finite states, declared counts,
StrawIds, foreign keys, parent closure and cycles. An actual saved-table run admits all 28,044 events and
1,376,129 SGS rows without filtering; source-wrapper presence remains a separate gate.

`response_presence.py` inspects the highest art Events cycle, exact SGS wrapper/object branch and boolean
presence for every event. It records physical and numeric-within-parent identity traversal, source hashes
and streamers. It does not decode SGS values or authenticate particle ProductIDs. Discovery rejects multiple
candidate products and freezes the exact discovered type/label/instance/process in its receipt.

`response_control.py` binds original export receipts and context, completed F injection evidence, the actual
frozen importer/dumper/runner bytes, exact original services/schedules/labels/seed policy and saved all-52
hybrid decision identities. It creates fresh parent tables/preparations without altering original exports.
Preparation alone is not numerical round-trip, physics or runtime validation.

`response_roundtrip.py` compares original and re-exported SGS by exact event identity and within-event
encounter ordinal. It checks frozen-dumper dtypes, all integer/float payloads and generator data at zero
numerical tolerance; no straw sorting, deduplication or coordinate/time rebasing is allowed. Zero-row event
coverage is reconciled through generator identities and must be combined with independent art presence.
Particle differences are separate non-gating diagnostics: the importer is deliberately not lossless.

Per-parent digitization and its read-only re-export use separate output directories. Postprocessing failure
must be repaired from successful saved art/table files rather than rerunning response. The all-parent
trigger preparation admits every parent before one continuous all-52 invocation with its own counters.
Actual consumed-particle ProductID and loaded-plugin identity remain explicit unresolved provenance gates;
executed exact configuration and frozen runtime bytes do not independently authenticate either.

### Observed context distinction

The original SGS export reads pre-digitization DTS: its observed product tag is
`compressDetStepMCs::Primary`. The retained regenerated stock context instead persists
`compressDigiMCs::Digitize`; it does not retain the old label. Both ten-parent populations are independently
presence-checked against the same original event ledger before preparation. Do not silently call these the
same collection or assume serialized SGS/pointer equivalence through production compression. C imports the
original exported pre-digitization SGS into the retained stock context as declared. C's read-only re-export
uses `compressDigiMCs::Digitize` for the retained generator collection and explicit injected SGS/particle tags;
the F importer/response producer construction itself remains unchanged. This adds a source/context distinction
to the existing importer/state/RNG caveats rather than removing one.
