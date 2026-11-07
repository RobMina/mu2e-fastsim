# Calorimeter numeric export satellite

This isolated satellite starts from the existing `r6_calo/calo/art` **CaloLibraryDumper**
exporter. No other analyzers are included and the original checkout is unchanged.
It exports saved input only; it does not generate or rerun transport.

## Explicit input modes and numeric additions

`inputMode` is a validated string: `stage` (default) or `primary`, with no auto-detection.
`CaloFastSim/fcl/dump_library.fcl` retains the pilot Stage input and analyzer-only job.
`dump_library_primary.fcl` includes that base and overrides the mode and production tags:
status `g4run::Primary`, transport/showers `compressDetStepMCs::Primary`, primary
`FindMCPrimary::Primary`, and **only** `compressDetStepMCs:virtualdetector:Primary` steps.
Other step collections are not assumed saved.

- `events` adds `input_mode` (int: 0 Stage, 1 primary), `transport_product_id` (uint32),
  `primary_present` (bool), `primary_product_id` (uint32), and `n_primaries` (uint64).
  These fields reset each event; absent products have ID/count zero.
- `primaries` records the actual `PrimaryParticle::primarySimParticles()` vector in order:
  `run`, `subrun`, `event`, `product_id` (uint32, the PrimaryParticle product), `index`
  (uint64), `sim_valid` (bool), `sim_product_id` (uint32), `sim_key` (uint64).
  Its product registry kind is -5; the registry size is the vector length.
- `particles` adds `self_parent` from `SimParticle::selfParent()` and `is_ce_endpoint`
  from comparison with `ProcessCode::mu2eCeMinusEndpoint` (both bool), preserving creation
  codes and parent/generator identities. Python need not hardcode this process enum.

The Stage export block is unchanged apart from its mode guard. Primary mode does not read
or fabricate StageParticles: `generated` stays empty and `generated_present`,
`generated_product_id`, `n_generated` remain false/zero. Stage mode leaves primary fields
false/zero. All primary multiplicities, including zero and multiple, are exported without
CE filtering; the Python acceptance gate must reject anything other than a single CE.
Every primary pointer goes through `requireSim`; on status zero its target product must
also equal the configured transport handle. Missing primary or transport on status zero
is a hard error. Nonzero failures remain diagnostic records, not valid empty responses.

Generator links are nullable: a modern CE may have a muon parent rather than a direct
GenParticle link. The existing `getMany<GenParticleCollection>()` export and exact
productID/index validation of every non-null generator link are retained. Missing
StatusG4 remains fatal; partial output from any failure must not be accepted.

## Opt-in immediate-source event identities

**User-built and validated on the complete, already-used 2,813-event development production file.**
All source IDs match the independent stock printer, and all 218 previous columns across 11 tables match the frozen
export exactly (same-position NaNs treated as equal). This validates the exporter, not cross-file independence.
`exportSourceEvents` defaults to false. With it disabled, no source lookup, source branches, source tree or registry
rows are added. `sourceEvents` defaults to an empty InputTag; enabling the export requires a nonempty label and
explicit process. The separate `dump_library_primary_sources.fcl` and `dump_library_sources.fcl` opt in with
`TargetStopResampler::Primary` and `TargetStopResampler::CaloLibraryCE` respectively. Existing FHiCL remains unchanged.
`export_primary.sh` uses the new production FHiCL only for `CALO_EXPORT_SOURCE_EVENTS=1` (default0; other values fail).

When enabled:

- `events` adds `source_present` (bool), `source_product_id` (uint32), `source_count` (uint64), `source_tag` and
  `source_branch` (strings). State resets every event. Missing sequence is fatal regardless of G4 status.
- `source_events` contains downstream `run/subrun/event`, sequence `product_id` (uint32), `index` (uint64),
  `source_valid` (bool), and `source_run/source_subrun/source_event` (uint64), in sequence order. Zero or multiple
  elements and invalid IDs are preserved as diagnostics; the CE reader requires exactly one valid source event.
- `products` adds kind **-6**, using the actual sequence ProductID, resolved branch name, presence and size.
  This is separate from kind-5 PrimaryParticle pointers. No shower/incident/state cut selects the provenance rows.

The EventIDSequence carries neither the original source filename nor a selected-muon key. These tables do not
invent either. Export one input file per invocation; associate the output with its verified input-file manifest.
Resolve the source filename from audited configuration/provenance only when unique. A copied pilot subset must
map back to its original source file; its new path/checksum does not establish a new independent parent.

`calofastsim.showers.source_events` checks the saved numeric schema, counts, sequence indices, ProductIDs and
registry against an **independently enumerated ordered full event census**, without any physics selection:

```bash
PYTHONPATH=calo python3 -m calofastsim.showers.source_events \
  --tables TABLES.root --expected-events EXPECTED_EVENTS.npz \
  --source-tag TargetStopResampler::Primary --source-branch art::EventIDs_TargetStopResampler__Primary. \
  --out SOURCE_EVENT_CHECK
```

The expected NPZ supplies integer `run`, `subrun`, `event` columns; do not derive it from the new export itself.
For the existing development file, a previously independently reconciled full event census can be reused.
The output is `source_events.npz` plus strict-JSON input/source receipts. All `source_file_resolved` flags remain
false; no FIDs or independence claim are manufactured. Source-file resolution, original-copy provenance and
manifest binding must still be verified before forming the ledger consumed by `source_split overlap`.
Missing opt-in fields in old exports are an error, never a fallback to downstream IDs. Discard partial output
from a failed art job; a valid-looking prefix is not a complete provenance export.

The extension includes the stock `MixTypes.h` EventIDSequence type declaration and uses existing Event/Handle and
canvas interfaces; no new linked-library calls or dependency changes are introduced. Both CMake and SCons retain
their existing art/canvas dependencies. The successful user build and completed art jobs establish build/runtime
success for this configured area; synthetic tests alone do not.

### Independent source-event runtime check

`CaloFastSim/fcl/print_primary_sources.fcl` runs the stock `EventIDSequencePrinter`, with one framework ArtReport
per output event and no reconstruction or transport. The checker binds each printed sequence to that ordered full
output EventID, requires a singleton, complete TrigReport and terminal art success, and compares every source ID.
The actual branch name is `art::EventIDs_TargetStopResampler__Primary.`; its namespace is not stripped.

```bash
PYTHONPATH=calo python3 -m calofastsim.showers.source_event_check census \
  --baseline CERTIFIED_OLD_TABLES.root --out EXPECTED_EVENTS.npz
PYTHONPATH=calo python3 -m calofastsim.showers.source_event_check validate \
  --baseline CERTIFIED_OLD_TABLES.root --tables TABLES.root --printer-log PRINTER.log \
  --source-tag TargetStopResampler::Primary --source-branch art::EventIDs_TargetStopResampler__Primary. \
  --out INDEPENDENT_SOURCE_CHECK
```

Every old table, column, dtype and row order must match; only new source fields/tree and registry kind -6 are
allowed. Packed awkward buffers compare jagged crystal neighbors without event loops or floating-point identity
conversion. ROOT inputs and printer log are hashed before and after validation; existing outputs are not replaced.
Baseline certification remains an external prerequisite, not a property established by copying its census.

On the checked development file, all 2,813 source IDs were valid singletons and matched the stock printer exactly.
Their 2,813 distinct immediate EventID tuples are a within-file diagnostic only, not proof of independence from
other production files or the legacy pilot. All 218 old columns in 11 tables matched. A repeated independent
source-art check also matched all output EventIDs and all 115,272 CSS rows/11 state fields, retaining the same 56
empty front responses and geometry checks. That exporter check inferred no source-file FIDs and produced no overlap
result. The subsequent separately authorized provenance-only audit below supplies verified file binding; neither
operation builds a new library, reconstructs events, generates Geant4 or accesses reserved confirmation files.

### Provenance-only training/development/pilot audit

`audit_parent_sources.sh` runs only the stock printer for the allocated training/development files after
`source_audit metadata` matches the frozen file manifest to SAM identity/size/checksums and unique original-parent
metadata. It uses a matching preconfigured work area and explicit analysis Python; it never builds. `source_audit
census` reads only EventAuxiliary from the highest Events cycle and hashes the full input against its catalog
SHA256. `bind` checks the complete ordered printer log, the single-file ResamplingMixer configuration, the independent
census receipt and its NPZ hash before resolving the original source FID. File0 reuses its completed printer log.
`print_pilot_sources.fcl` supplies the corresponding stock-printer tag for the retained legacy pilot.

`source_pilot_audit` binds the pilot artifact through its saved input/config receipts to the original copied stop
pool, requires its actual source sequence to equal the full independently decoded copy census, and checks old-export
EventID order plus the frozen library's donor-row identities. Its SHA256-based output identity is explicitly an
artifact identity, **not a SAM FID**. Both binders hash their generated expected/ledger arrays in the final receipt.
`source_overlap_report` requires these hashes and complete counts before comparing original-file/EventID keys.
`source_audit rebind` regenerates authenticated bundles from saved proofs only, without art rereads or new sampling.

The completed allocation has 28,265 training events, 27,944 development events and 2,000 pilot events. Training shares
zero immediate parents with development or pilot; development shares 165 with pilot, all in sequencer200, including
40 of the frozen505 donor parents. File0/full-pilot overlap is zero. No rows were silently pruned or reassigned.
See [the scoped result](../SHOWERS.md#completed-bounded-original-parent-audit): this is not original-proton
independence, a confirmation result, inclusive acceptance, or a new shower-coverage certificate.

### Training-only coverage export

The separately authorized ten-file training census is complete: 28,265 stored events, 24,120 direct-CE front
entries, all status zero. It retains 259 empty and 835 weak (whole-event Birks ≤10 MeV) front responses and fills
55 of the unchanged 96 disk/energy/cosine/radius cells; the frozen505 library fills 25. See
[the coverage result and limitations](../SHOWERS.md#training-only-selected-production-coverage-census).

`export_training_coverage.sh` reads only the authenticated fixed training inventory, sequentially, using the
existing user-built exporter with source-ID opt-in. Each input's saved full GeometryService table must equal the
verified production reference before extraction. Original-release geometry search paths remain mandatory. It
checks input SHA256 before/after, keeps a complete status diagnostic, rejects nonzero statuses or incomplete
allocated counts, and independently reconciles raw CSS plus geometry. It does not create a library or certify
source binding merely by finishing: `training_coverage` must also check the saved provenance/receipt chain.

The local report takes an externally frozen portable extraction hash manifest and completed parent-audit summary.
It checks every new output/source ID against saved independent census/printer proofs, their authenticated final
parent binding, all extraction artifact hashes, saved geometry settings and numeric frames. Both-disk whole-event
CSS totals retain zeros; non-front and out-of-bin populations remain explicit. Only per-cell arrays and reports are
written, never packed donor patterns. Original raw-evidence authentication is a prerequisite, not a signature
inferred from self-consistent hashes. The complete raw CSS comparison matched 1,158,498 rows; no new development
physics or confirmation inputs were read. Defaults, K8 and the legacy library are unchanged.

### Later production-only candidate packing

A separate authorization subsequently packed the completed training exports into one candidate library, using
all 24,120 valid front entries, zeros included, with no legacy donors. This is local numeric packing through the
existing Python library builder, not a software build or new art job. Donor provenance carries the complete
output/source identity and stable file/global/local event coordinates; whole-event CSS and geometry round trips
are checked against the authenticated coverage census. `production_candidate` writes new library/provenance/
metadata artifacts and hashes, never overwriting the legacy library.

`candidate_support` then audits deterministic K8 pools on the already-used file0, without sampling/reconstruction.
It checks the full query source ledger and original-parent disjointness, candidate outputs and embedded metadata,
all donor coordinates against authenticated training boundaries, geometry and exact historical legacy-pool parity.
All 2,396 front queries are supported; fallback changes 73→1. See [the result and limits](../SHOWERS.md#separate-production-candidate-and-deterministic-k8-support).
No additional export, build, development-file physics read or confirmation access was required.

The later separately authorized [fixed production-candidate trial](../SHOWERS.md#fixed-production-candidate-reconstruction-trial)
used those same saved file0 tables for one shared draw and exactly two CPU reconstruction calls. All stock/control/
legacy references were reused. It required no exporter change, new art read or compiled build; reserved files remain
closed. Selection improved substantially, but this is selected-population development, not inclusive closure.

## Build and geometry gates

**The earlier exporter was user-built and checked on one development production DTS file; not independent physics validation.**
The analyzer-only smoke and complete-file extraction both exited successfully. All 2,813 stored event identities
and 115,272 ordered CSS rows matched an independent source decoder exactly (including persistent pointers and
all exported CSS state fields). Identity/ancestry/count checks passed; 56 empty responses were retained.
The calorimeter tables contain two disks with 674 crystals each; all 9,221 saved front-VD crossings lie within
the corresponding exported slabs. These checks do not certify every particle/step field or full detector material
parity. The confirmation files remain unused.
The minimal CMake and SCons manifests list only the original library dumper dependencies.
`CaloFastSim/.muse` is the required, distributed repository-discovery marker; without it
muse skips the entire exporter and can report a successful no-op build. The work-area
`art/.muse` only selects the envset and does not replace that marker. `setup_area.sh`
requires both the marker and `src/SConscript` before preparing the area. Work-area `.muse`
and backing choices are ignored, not distributed source. Prepare an explicitly pinned
coherent backing/envset using:

```bash
CALO_BACKING=/absolute/pinned/backing CALO_ENVSET=pNNN bash calo/art/setup_area.sh
```

That script only prepares the area and prints the user build command. The user must perform
a full `muse build -j 32` on the designated host, not a package-target build, and verify
headers/linkage before any separately authorized execution.

The primary FHiCL inherits pilot geometry filenames; **filenames alone do not establish parity**.
For the checked sample, its saved job configuration names Offline v13_41_01 and the same Run 1a geometry and
field files. `export_primary.sh` prepends that original release to `MU2E_SEARCH_PATH`, preserving its include
closure and crystal-position file rather than silently taking the newer backing's definitions. Source comparison
found identical CalorimeterGeom, only added material-name storage in DiskCalorimeterMaker, and no changes to the
calorimeter VD placement block. Exported local/global crystal frames, disk-front VD positions and saved crossings
were checked numerically. This is coordinate parity for this sample, not a full-material certification.

`export_primary.sh` requires the already configured area, a fresh plugin, an explicit existing development input,
event limit, fresh output directory and original simulation search root. It never builds or runs Geant4. For a
new sample, repeat the saved configuration/release check; do not reuse these conclusions by filename alone.
After a completed whole-file extraction, reconcile it against the original art without resampling:

```bash
PYTHONPATH=calo python3 -m calofastsim.showers.production_check \
  --art SOURCE.art --tables TABLES.root --check-geometry --out source_check.json
```

The independent CSS decoder's ProductID offset is +18 bytes within the persisted pointer (three six-byte
headers), not +16. The exact source check exposed that two-byte error; the corrected decoder has a regression
assertion for ProductID. Response/reconstruction did not consume that identity column, and this fix does not
change its energy, time or position inputs. No new transport, library tuning or acceptance claim follows.
