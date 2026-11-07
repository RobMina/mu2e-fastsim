# Read-only historical aggregated board export

`FlashBoardExporter` is a validated-FHiCL EDAnalyzer in a new isolated muse
satellite. It reads **already aggregated** `StrawGasStepCollection`,
`CaloShowerStepCollection`, `CrvStepCollection` and SubRun `GenEventCount` products.
It neither changes event products nor schedules transport, aggregation, filters,
compression, RNG use or a production-default change. The raw R5 satellite and
frozen raw references are separate and untouched.

Read the [board observable](../../docs/flash-board-observable.md) and
[historical selector](../../docs/flash-board-selection.md) before interpreting
these tables. This is extraction, not a fast prediction or physics closure.
Historical selected anchor art can check retained-stream arithmetic; it cannot
recover or validate detector-rejected events absent from that stream.

## Backing and validation status

The intended reader backing is **Offline v13_39_00 / envset p107**, matching the
historical analysis era, not today's Offline production defaults. The enclosing
`beam/art/.muse` pins the envset; **`FlashBoard/.muse` is also required** to register
this repository with Muse. Only then does SCons discover `src/SConscript`, which
uses the existing FlashReference satellite's direct dependency list. There is
no separate CMake build system or private Offline checkout.

The initial user build did no work because the repository-level marker was
missing. It is now supplied; a fresh shell and `muse setup` are required to
refresh discovery. Before building, `MUSE_LOCAL_REPOS` must contain `FlashBoard`,
and `MUSE_WORK_DIR` must be the enclosing `beam/art` area. An "up to date" message
or `.musebuild` timestamp alone is not proof of compilation. Expected plugins are
`build/al9-prof-e29-p107/FlashBoard/lib/libflashboard_FlashBoardExporter_module.so`
and `libflashboard_FlashBoardFixture_module.so` in the same directory.

Both compiled plugin libraries and object files were subsequently observed with
timestamps newer than their sources and a completed `.musebuild` stamp. The build
log and runtime/plugin identity still need verification; **no FHiCL expansion,
mu2e execution or ROOT/art reread has been validated.** Product
constructors and accessors were inspected in the local `remote/off/Offline`
headers; art provenance, TFileService and fixture patterns were compared with
the existing FlashReference satellite, and end-SubRun counter production with
`EventGenerator/src/GenEventCounter_module.cc`. This does not establish binary
or historical-release API compatibility. In particular the historical backing's
ROOT float-vector `R()` return type is deliberately checked by `static_assert`:
a different type must fail compilation, not be repaired with a cast or a new
double-precision norm. End-SubRun producer/analyzer ordering, persisted pointer
resolution, FHiCL behavior and output-file boundaries still need a user-approved
built-fixture/reread check.

The user, not the agent, performs a full `muse build -j 32` on the designated
build host after the enclosing area and backing are reviewed. Do not run the modules until
the user confirms that build and its runtime/plugin identity have been verified.

## Configuration

Use analyzer label **`board`** to obtain ROOT directory `board/` (TFileService
uses the module label). The validated parameters are:

- `strawSteps`, `caloSteps`, `crvSteps`: nonempty vectors of mandatory InputTags.
- `generatedEvents`: one mandatory InputTag for a SubRun `GenEventCount`.
- `expectedEvents`, `expectedSubRuns`: positive exact `ULong64_t` counts.

The `export.fcl` prolog leaves tags unset and expectations zero; `read.fcl`
also defaults `source.maxEvents: 0`. It intentionally refuses to start until a
wrapper supplies the exact tags and counts. Supply input filenames and a fresh
output path too. A process-qualified InputTag is preferable when provenance
requires one. Unqualified tags are recorded as configured and the actually
resolved process is recorded independently.

Do not infer historical tags from the names used in this fixture. Local stock
`Analyses/test/edep.fcl` mentions `compressDetStepMCs` for straw and calo, but this
is not a census of the historical files or a confirmation of their CRV and
counter branches. Therefore no historical detector/counter tags are invented in
`read.fcl`: inspect the saved input product inventory/configuration and explicitly
set all four inputs. Multiple collections per role are supported and all must
be declared, even when empty.

## Schema v1

The following is the exact contract shared with `beamfastsim.flash_board_export`.
`u32`, `u64`, `f32` mean ROOT `UInt_t`, `ULong64_t`, `Float_t`; all text is
`std::string`. Identities never pass through floating point. No additional columns
are required.

| Tree | Columns |
|---|---|
| `schema` | `version:u32=1`, `expected_events:u64`, `expected_subruns:u64`; exactly one row |
| `requirements` | `detector:u32`, `slot:u32`, `tag:string`; one row per configured collection |
| `events` | `run,subrun,event:u32`; `nproducts,nstraw,ncalo,ncrv:u64`; one row per fully exported event, including all-empty events |
| `products` | `run,subrun,event:u32`; `product:u64`, `detector,slot:u32`, `size:u64`; `tag,label,instance,process,branch:string`; one row per configured collection per event, including empty collections |
| `straw_steps` | `run,subrun,event:u32`; `product,index,particle_product,particle_key:u64`; `ionizing_mev,momentum_mev_c:f32` |
| `calo_steps` | Same identities; `birks_mev,momentum_mev_c:f32` |
| `crv_steps` | Same identities; `visible_mev,momentum_mev_c:f32` |
| `subruns` | `run,subrun:u32`; `product,generated_events:u64`; `tag,label,instance,process,branch:string`; one row per successfully read end-SubRun counter |
| `completion` | `complete:u32=1`; `events,subruns,straw_steps,calo_steps,crv_steps:u64`; exactly one row only at successful exact-count endJob |

Detector codes are tracker **1**, calo **2**, CRV **3**. Slots start at zero
independently within each configured role and preserve configuration order.
`tag` is `InputTag::encode()`; label, instance, resolved process and branch name
come from the actual product provenance. `product` is the actual art ProductID,
not the configured slot. `index` is the original collection index. Particle
identity is the resolved SimParticle Ptr's product **and** key; equal keys in
different products remain distinct particles.

Energies are exactly `ionizingEdep()`, `energyDepBirks()`, `visibleEDep()`.
Momenta are exactly `StrawGasStep::momentum().R()`,
`CaloShowerStep::momentumIn()`, `CrvStep::startMom().R()`; all six return types
must be `float`. There is no reconstructed double norm, threshold, PDG/time mask,
selection or photon weighting in the exporter.

Every event's handles, provenance, pointers and accessor values are validated
before any of its payload is written. The event ledger is filled last. Null,
unavailable or non-resolving SimParticle pointers throw; lookup exceptions are
not caught and converted to zero. Nonfinite/negative energy or momentum,
missing products/provenance, duplicate per-role tags, aliases to an actual
ProductID across configured collections, duplicate EventIDs and count overflow
throw. Every `TTree::Fill()` is checked for a negative result. A failure latches
in memory and prevents an endJob completion row, even if framework cleanup
invokes endJob. Write failures can leave partial payload: consumers must refuse
an absent/invalid completion marker, not try to salvage it as a valid export.

`endSubRun` uses `getValidHandle<GenEventCount>` on the configured tag with an
`art::InSubRun` consumes declaration. Zero and the unsigned-long default sentinel
are rejected. Duplicate SubRun callbacks throw rather than summing fragments.
Exposure is never inferred from detector events or acceptance. Events and
SubRuns exceeding expectations throw; endJob requires exact equality and a
counter for each event's SubRun.

**A completion row does not certify full-sample exposure.** Caller-supplied
counts can describe a truncated input too. The caller must supply all relevant
files/events for every counted SubRun, including SubRuns with no retained
events, and separately establish successful job completion, closed-file
integrity and upstream transport validity. No persistent receipt, watchdog,
transport-cap or scheduler framework is supplied here.

## Deterministic fixture (no G4)

`fixture.fcl` schedules `FlashBoardFixture`, the exporter and unfiltered
`RootOutput`; all products are saved. EmptyEvent generates four events in run 1,
two each in SubRuns 1 and 2. Rows below are in source order; the producer uses
that ordinal, not an assumption about event-number reset at a SubRun boundary.
The fixture refuses a different sequence or an incomplete run.

| Source ordinal (1-based) | SubRun | Tracker ionization (MeV) | Calo Birks (MeV) | Historical detector OR |
|---|---:|---|---|---|
| 1 | 1 | `float32(2e-6)` from A:17, `float32(5e-7)` from B:17 | empty | retained by tracker |
| 2 | 1 | `float32(5e-7)` from A:17 | `float32(.02)` from B:17 | retained by calo |
| 3 | 2 | empty | empty | rejected |
| 4 | 2 | `float32(5e-7)` from A:17 | empty | rejected |

The first event also has one CRV step with visible energy `float32(2e-6)`,
particle B:17, and start momentum `(3,4,0)` giving exact accessor `5.0f` MeV/c.
Its CRV branch accepts the already tracker-accepted event. All other CRV
collections are explicitly empty; the second SubRun still has no retained event.
Tracker/calo stored momentum accessors are exactly `1.0f` MeV/c. Both `fixture:particlesA` and
`fixture:particlesB` contain SimParticle key **17** in every event; their actual
ProductIDs differ. The three detector tags are `fixture:tracker`, `fixture:calo`,
`fixture:crv`, each at slot zero. `generatedEvents: "fixture"` reads counters
**10** and **20**, produced at endSubRun with `art::fullSubRun()`. The second
SubRun has no retained event but contributes exposure 20.

Expected export: **4 events, 12 products, 4 straw rows, 1 calo row, 1 CRV row,
2 SubRuns**, three requirement rows and one schema/completion row each. Per-event
`nstraw=[2,1,0,1]`, `ncalo=[0,1,0,0]`, `ncrv=[1,0,0,0]`, `nproducts=3`.
The retained mask is `[true,true,false,false]`. The numerator is
`float64(float32(2e-6)) + 2*float64(float32(5e-7))`, approximately `3e-6` MeV,
including both retained subthreshold tracker deposits. The denominator is **30**,
not 4 or 2; Q is approximately `1e-7` MeV/generated event. All four exported
tracker steps instead sum to approximately `3.5e-6` MeV. Treat this fixture as
`filter-input`, not an already retained stream, when evaluating the selector.
There is deliberately no second calo collection in this minimal fixture.

`fixture.fcl` writes `flash_board_fixture.root` and
`flash_board_fixture_%#.art` (Job granularity; inspect the resulting closed-file
inventory). `read_fixture.fcl` reads the saved art without any producer and uses
the same tags/counts, writing `flash_board_fixture_reread.root`. After a verified
user build, a separately approved check should compare all typed payload rows
and resolved provenance between initial export and saved-art reread, and check
the analytic expectations above. The independent checker is:

```sh
python3 -m beamfastsim.flash_board_fixture --input flash_board_fixture.root \
  --reread flash_board_fixture_reread.root
```

It checks both fixtures individually before comparing every typed payload field.
Always choose fresh paths and refuse overwrite.

Negative configurations:

- `fixture_bad_ptr.fcl` sets `badPtr:true`: the first tracker step points to
  nonexistent key **999** in particlesA. It must throw during pointer resolution
  and produce no completion row; an unrelated startup failure is not a pass.
- `read_missing.fcl` reads fixture art but requires `fixture:missingTracker`.
  Missing product lookup must throw, never create a synthetic empty collection.

Also test wrong expectations, duplicate/aliased tags and duplicate event/SubRun
identities after a build is available. None of these configurations has been
executed as part of this source-only change.
