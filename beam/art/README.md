# Read-only raw reference exporter

`FlashReference/` is an isolated muse package beside the Python beam code. It
links against a declared existing Offline backing, not a private Offline checkout.
`FlashReferenceExporter` is an EDAnalyzer: it reads normal art handles, writes
TTrees through TFileService and never changes event products, Geant4 or RNG state.
`FlashReferenceFixture` is a deterministic EDProducer for extraction tests only;
it must never appear in a transport configuration.

See the [export contract](../docs/flash-reference-export.md) and
[raw reference scope](../docs/flash-reference.md). Source inspection and Python
schema tests do not prove compilation or successful art execution. Do not launch
transport merely because the package directory or a completion tree exists.

## Pause checkpoint — 2026-11-05

The user completed the isolated build; source/plugin hashes are recorded in
[the R5 handoff](../docs/r5-pause-20261105.md). Confined four-event/two-SubRun
synthetic extraction, saved-art reread and independent saved-output checks
passed in their stated scope. This does not certify the complete transport
runtime, descendant cleanup, aggregate output bound or photon flux. No rebuild
or transport launch is requested by this archive. Instructions below remain a
reference for preparing a fresh area, not an outstanding build request.

## Prepare and build

From this directory, `bash prepare_area.sh EXISTING_OFFLINE_BACKING` creates only
the `backing` link and refuses to replace a different existing link. The area's
`.muse` pins envset p109 for the declared release. The script contains
no environment setup, build or submission. The backing must match the declared
raw-reference runtime. It does not clone, modify or rebuild Offline.

The user performs the full build on the designated build host after source
review, from this `beam/art` area:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
setup muse
muse setup
muse build -j 32
```

Wait for a successful build and verify plugin/source timestamps, runtime stub,
backing, resolved libraries and `MUSE_WORK_DIR` before running either module.
`src/SConscript` lists plugin dependencies directly because scons link dependencies
are not transitive. This satellite has no independent CMake build system.

## Schema v1

The analyzer label determines the ROOT directory, conventionally `export/`.
Numeric fields retain persistent precision. Integer IDs do not pass through float.

| Tree | Scope and content |
|---|---|
| `schema` | One row: schema version, native ProductID and Ptr-key byte widths |
| `storage` | Machine-readable persistent-to-export representation groups, including widened IDs, short ion level and bool flags |
| `requirements` | Configured mandatory tag/kind, exact positive expected-event count and ordinal base |
| `events` | One per successfully exported event, full EventID, declared ordinal and totals, including empty events |
| `products` | One row per required step/status product and discovered particle/generator collection per event; one per configured SubRun volume collection; full branch, friendly type, actual framework `produced_type`, declared collection `type`, module/instance/process, ProductID, size and configured role |
| `particles` | Every available SimParticle collection, key/id/stage and all persistent particle states/codes/ion detail and parent/generator Ptrs |
| `daughters` | Ordered daughter Ptrs with owner product/key; `index` is daughter ordinal within that owner |
| `steps` | Every row of every configured StepPointMC collection; pre/post state, double pre-times, stored float deposits/length, raw volume number and numeric process |
| `generators` | Every available GenParticle collection, complete stored state and generator ID |
| `statuses` | Every configured StatusG4: status, tracks, overflow, step/field kills, CPU and real time |
| `stages`, `volumes` | Full SubRun product identity; explicit stage counts including empty stages; volume table keys distinct from copy numbers, names/materials |
| `completion` | endJob marker and exported event/SubRun counts; not a scheduler success or closed-file certificate |

Common event columns are uint32 `run/subrun/event` and uint64 `ordinal`;
collection rows additionally carry uint64 `product/index`. `index` is row ordinal,
except `volumes.index` is the map key and `stages.index` is the stage index.
SubRun rows have `event=0, ordinal=0` and `products.scope=1`; those zeros are
scope markers, not real EventIDs. All event product rows have `scope=0`.
`products.kind` is 1 particles, 2 generators, 3 steps, 4 status, 5 volumes.

Ptr columns use uint32 `*_state`, uint64 `*_product` and `*_key`. State 0 is null,
1 available and dereferenceable, 2 nonnull/unavailable, 3 resolution exception.
Raw product/key values are retained even for null Ptrs; inspect state rather than
a numerical sentinel. Resolution errors are recorded and counted in the event
ledger, never silently made null. They refuse completeness in the auditor, not
transport output selection. No pointer cache address or transient getter is exported.
Null parent/generator links can be valid; a null step-particle or daughter link is
not a complete reference. The analyzer does not recursively follow ancestry.

Missing mandatory products throw. An existing empty product is always exported
as a zero-sized product row. No optional catch creates empty products. Exact duplicate
configured tags, aliased step/status/volume tags and duplicate EventIDs throw.
All available particle/generator collections are discovered so full Ptr closure
can be audited, while configured required lists independently demand source and
continued products. Event-level volume products are not silently substituted for
SubRun volumes: configurations using those need a separate explicit schema extension.

## Safe configurations

- `FlashReference/fcl/read_source.fcl`: existing source catalog only, required
  source steps/particles/generators/SubRun volumes; no downstream status invented.
- `FlashReference/fcl/read_raw.fcl`: existing saved raw reference, required photon,
  tracker, calo, CRV, VD steps and input/continued particles, statuses and volumes.
- `FlashReference/fcl/fixture.fcl`: four synthetic events over two SubRuns, empty
  events/collections/stage, identical particle keys in distinct products, copied
  and new stages, repeated source hits, large double time, nonzero post-state,
  all deposits/processes, null optional ancestry and status fields.

Both read configurations default to `maxEvents: 0, expectedEvents: 0`. The exporter
refuses a nonpositive expectation at construction, events beyond it and an endJob
count that differs from it. Supply input filenames, event count, analyzer
`expectedEvents` and output path explicitly; set `ordinalBase` to the declared
source skip. That value labels the assignment; it does **not** skip the RootInput
reader or prove source pairing. Verify actual EventIDs and states after reading.

After the user build is verified, use a fresh output directory for each command
and refuse overwrites. Example commands within the configured area, with every
output directory already chosen in the project's data space:

```bash
mu2e -c FlashReference/fcl/fixture.fcl -T fixture.root
mu2e -c FlashReference/fcl/read_fixture.fcl -s flash_reference_fixture_1.art -T reread.root
mu2e -c READ_SOURCE_WRAPPER.fcl -s INPUT.art -n 1000 -T source.root
mu2e -c READ_RAW_WRAPPER.fcl -s RAW.art -n 10 -T exported.root
```

Use `python3 -m beamfastsim.flash_reference_fixture --input fixture.root --out fixture_check.json`
for the independent analytic fixture expectations, then repeat on `reread.root`.
All status fields are checked exactly; an expected injected failure cannot be
satisfied by an unrelated failure. The analytic checker is specific to this fixture,
not a substitute for full independent arbitrary-product extraction comparison.

The fixture writes `flash_reference_fixture_1.art` in the working directory.
Its nested `fileProperties` table requests Job granularity; RootOutput requires
`%#` in the filename whenever that table is supplied. The counter does not alone
prove a one-file inventory; inspect the closed output and its job log.
Re-read that saved file with fixture tags in a read-only wrapper (no producer)
and compare all required exported payload rows and dtypes exactly:

```bash
python3 -m beamfastsim.flash_reference_compare --left fixture.root --right reread.root --out comparison.json
```

The comparator assesses payload equality only, not validity; identical failures
must still fail the separate auditor. Exercise
`physics.producers.fixture.failStatus:true` and `unavailablePtr:true` in separate
fresh fixture jobs, and a missing required tag in the reader: no failure is to
be filtered or converted to a passing empty reference. Test repeated source
EventIDs deliberately to confirm refusal. A complete first transport check still
requires source pairing, stage/parent correspondence, cap/warning/status audit,
closed-file/job success, field/runtime identities and the hard aggregate output
guard. None of those is supplied by a synthetic fixture or endJob marker alone.
