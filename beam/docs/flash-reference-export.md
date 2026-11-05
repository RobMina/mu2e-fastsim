# Raw reference export contract

The [isolated art package](../art/README.md) implements this read-only analyzer
contract. A verified build and the extraction controls below are required for
each declared runtime; source presence is not permission to bypass the raw-reference launch gates. It uses
normal art product access and ROOT TTrees readable with uproot. It does not modify
transport, material configuration, data-product schemas, cuts or random draws.

## Pause checkpoint — 2026-11-05

R5 is paused. Photon births and geometry helpers are not validated exit flux; S2,
the 66-point comparison and tracker/CRV occupancy validation remain unfinished.
No owned Geant4 pilot or production-slot use has occurred. The separate pidfd-free
controller has 114 mocked tests but no host launch; its outer watchdog and
invocation-bound receipt are unimplemented. The latest combined focused software
checkpoint was 2029 passed / 10 skipped (overlapping earlier suites, not additive
or a full physics validation). See [the handoff](r5-pause-20261105.md),
[output/lifecycle gates](flash-reference-output-bound.md) and
[owned-child design](flash-reference-owned-children.md). Nothing resumes automatically.

## Existing interface audit

| Interface | Useful evidence | Why it does not close this contract |
|---|---|---|
| `StepPointMCDumper` | Stored pre-state and optional double KinKal particle state | No full event identity, post-state, product-qualified Ptr or complete deposit/process fields; no empty-event row |
| `SimParticleAnalyzer` | Retained ancestry diagnostic | Float ntuple, no SubRun, ProductID or simulation stage |
| EventNtuple MC summaries | Full event header | Reduced float MC states and incomplete post-state/Ptr information |
| `PrintModule` | Event/SubRun scaffolding and volume diagnostics | Explicit fixed precision; missing stage/Ptr/step fields; status printer omits field-propagator count |
| `StepPointsPrinter` | Event identity, count and streamed step | No round-trip precision control, narrowed printed key, no complete particle/status complement |
| `StrawGasStepTableDumper` | Event-qualified detector summaries | Aggregated detector products required; partial VD and ancestry schema, not full raw StepPointMC export |

This inspection is source evidence, not certification of an installed release.
Any other existing plugin must be checked against its exact source and binary;
plugin names or successful configuration expansion alone are insufficient.

## Required tables

Every event table uses integer run, SubRun and event IDs. Every collection has an
explicit provenance entry: branch name, module/instance/process tag, ProductID and
persistent type. A particle identity is `(run, subrun, event, ProductID, key)`;
nullness is explicit and not encoded by casting a sentinel into an unsigned key.
Use integer identity columns without narrowing through floats.

- **Event-qualified coverage:** `events` has exactly one row per input event, even
  if every recorded collection is empty, with source ordinal and aggregate totals.
  Join event-qualified `products` rows for required-product presence and collection
  sizes, and `statuses` rows for complete StatusG4 fields: status, number of tracks,
  SimParticle overflow, step-limit and field-propagator kills, CPU and real time.
  Missing required products throw; never synthesize an empty collection. This
  analyzer is not a filter, and failing events remain in the denominator ledger.
- **Source hits:** ordered StepPoint indices, originating collection and particle
  Ptrs, copy number, pre/post position and momentum, global/proper time, all three
  deposit fields, length and end-process numeric code. Preserve the input
  SimParticle collection separately so source ancestry is not reconstructed from
  approximate time or spatial matching. Export every species and repeated hit.
- **Particles:** input and new-stage collection identities, collection key and
  stored particle ID, simulation stage, full parent/generator/daughter Ptrs,
  PDG code, creation/stopping codes, start/end positions, momenta, times, volume
  indices, G4 statuses, last-step/end kinetic energies, step count, track length
  and stored ion detail. Copied ancestors are not new births. Preserve daughter
  ordering and collection order; never infer the new stage from maximum key alone.
- **Steps:** every configured photon/tracker/calo/CRV/VD collection, including
  explicit zero counts in Events. Include ordered row index, full particle Ptr,
  copy number, pre/post vectors, double pre-step global/proper times, stored
  total/nonionizing/visible deposits, length and end-process code. Coordinates
  remain in the product's documented frame; no implicit transformation.
- **Generators:** complete stored state and generator ID, keyed by collection and
  row index so generator Ptrs can be checked without relying on a display name.
- **Volumes:** full SubRun/product identity, each simulation-stage index, table
  key, copy number, name and material. A step copy number is not a unique table
  key. This table is not a solid/touchable-path or online crossing record.

## Precision and comparisons

Export stored integer/float/double types without further loss. Widened float
columns may be convenient but must retain an explicit storage-type declaration.
Persistent SimParticle times and momenta are floats; their original Geant4 double
values cannot be recovered by an exporter. Source StepPoint times are doubles.
Check new-primary times/momenta against the exact destination-type conversion of
the source states, not against an arbitrary loose numerical tolerance. Exact
source replay pairing compares the original source products, not those converted
new-primary values. Position and momentum endpoint semantics must follow the
actual installed producer, including the saved pre-post-step-doit terminal state.

## Validation before transport

A user-built isolated analyzer, if needed, requires its own successful build and
runtime verification before use. Validate against existing saved art without new
transport: repeated keys in different products/SubRuns, empty collections, copied
and new-stage particles, repeated source hits, missing products and large times.
Use both hand-constructed table fixtures for analysis tests and an independent
persistent-product comparison for extraction; self-consistency of a table is not
proof that all source fields were extracted. An unsupported field fails explicitly.

Before reviewing the first transport checks, require closed files, exact saved
EventID/count agreement, identical ordered source assignments across designs,
source-to-primary parent/state correspondence, complete Ptr resolution and
particle/step cap headroom. Photon recorder saturation fails even if StatusG4 is
zero. Keep status/warning failures; do not renormalize a survivor subset.

Post-step times and native closed-boundary crossing flags are absent from these
raw products. Their reconstruction remains a separate, ambiguity-aware analysis;
this exporter must not fabricate them or certify exact online crossings.

## Implemented analysis interfaces

`beamfastsim.flash_reference_export` reads highest-cycle TTrees with exact branch
dtypes, validates explicit expected-event/requirement/storage metadata, ledgers,
collection sizes/order, complete event/product/key Ptr targets, daughter ordering,
status fields and explicitly selected cap headroom. Null Ptr state is separate
from its retained raw ProductID/key, including framework null sentinels. Failed
status/unavailable references remain visible; no event is removed or renormalized.
`compare_source_exports` compares exact selected exported source products/rows;
that is not an independent check of the original art extraction. Source particle
and generator comparisons need their explicit product selection, not just the
source-step comparison. The requirements table is a declaration, not independent
proof that the job configured every necessary product.

CLI audits create a new JSON file exclusively and return nonzero for failures.
For example, after a verified art read in a fresh data directory:

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.flash_reference_export \
  --input exported.root --expected-events 10 --out audit.json \
  --photon-role g4run:r5PhotonSteps --photon-cap 1000000 \
  --particle-tag g4run::flashReference --particle-cap 1000000
```

A successful internal audit is deliberately labelled `physics_completeness:false`:
external closed-job, runtime, extraction, source-to-primary, warnings and geometry
checks are still required. Missing optional cap arguments remain unchecked, not
implicitly safe. Raw art is authoritative throughout. The deterministic fixture
checker `beamfastsim.flash_reference_fixture` adds independent analytic value
expectations for the synthetic fixture only, with no transported-physics claim.
`beamfastsim.flash_reference_compare` checks all required table payloads in stored
order with exact numeric dtypes/bytes and logical strings; equality does not imply
validity. NumPy Unicode widths and object-versus-Unicode containers are incidental
loader choices, not stored ROOT string types; logical values remain exact. Run the separate auditor even when a deliberately invalid fixture is
unchanged by persistence.

`beamfastsim.flash_reference_source_check` compares a source export with an
independently produced, explicitly SHA256-pinned `flash_step_input` states NPZ.
It checks ordered event identities/counts, particle ProductID/key, volume copy,
joined PDG, exact destination-float32 pre-position/momentum bytes and original
double time bytes. It does not certify unmeasured post-state, proper time,
deposits, process fields or complete particle/generator extraction. No tolerance,
time folding, event rejection or provenance alias fallback is applied. Supply
`--input`, `--states`, `--states-sha256` and a fresh `--out` path; the standard CLI
requires the declared 1000 events and 1130 rows.

`beamfastsim.flash_reference_partition` checks that first/remainder exports exactly
partition the full ordered assignment, with contiguous original ordinals, complete
event payloads and consistent repeated shared-SubRun products/stages/volumes.
The standard CLI takes `--full`, `--first`, `--remainder`, `--out` for a 10/990
split of 1000 source events. `ordinalBase` does not skip input; the remainder
reader independently sets `RootInput.skipEvents: 10`. This certifies exported
assignment/payload agreement, not transported source-to-primary correspondence.
