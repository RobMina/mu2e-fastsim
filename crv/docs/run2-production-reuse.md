# Reuse Run 2 production before commissioning more transport

New surface-transport production is on hold. A seed/output collision check inside a personal campaign is not a
production-availability audit. SAM currently contains a substantial v10 candidate:
`sim.mu2e.CosmicDSStopsCRYAll.MDC2030af.art` (17,841 files, 1,902,797,992 retained events, queried 2026-10-29).
Its exact production configuration selects SimJob MDC2025af / Offline v13_00_04, `S1DSStopsCRY.fcl` and
`geom_run2.txt`. The top-level geometry, v10 CRV counters and default CRY input match the later reference bytes.
This does not prove transitive geometry, physics or field equivalence. The original S1 log explicitly selects
`bfgeom_v01.txt` and `ShieldingM`, whereas the full-reference worker uses `bfgeom_no_field.txt` and
`ShieldingM_EMZ`; both report Geant4 11.3 patch02. These are concrete comparison axes, not interchangeable
transport configurations. The one-event source probe validates the expected products and pointer identities;
full-file persisted census and exposure checks remain separate requirements.

## Validated one-file extraction

Production shard `001500_00000517` has now been extracted without transport. Its catalog-bound source SHA256 is
`519581828b3ba1797ee79e3b75781a8b8510a84333f178e5e14dcde76d3a6fff`. The highest-cycle census contains all
106,796 retained events; per-event generator/particle/step row counts, run/subrun/event bounds and strict art
pointer checks pass. Persisted livetime is 29.5617637634 s, consistent with the original log's rounded 29.5618 s
for 2,500,000 generated events. The extracted tables contain 108,612 generator rows, 1,518,175 particle rows and
1,505,793 converted CrvSteps. All 23 configured sectors have primary-muon crossings, including sides, U/D and C.

| Sector group | Primary-muon event/bar crossings | Contributing retained events | Conditional mean-length marginal SE |
|---|---:|---:|---:|
| R1–R6 | 44,049 | 9,344 | 0.276% |
| L1–L3 | 34,635 | 7,431 | 0.307% |
| T1–T5 | 427,190 | 91,425 | 0.067% |
| E1–E2 | 1,559 | 311 | 1.781% |
| U | 1,068 | 228 | 1.642% |
| D1–D4 | 11,470 | 2,177 | 0.494% |
| C1–C2 | 1,974 | 418 | 1.007% |

These are observed single-file, S1-conditioned statistics, not an all-generated crossing census or a production
coverage correction. Individual sectors can remain sparse: D4 has 112 crossings from 24 retained events, and C2
has 174 from 47. The wider dataset has not been extracted or assigned a verified generated exposure. Availability
and safe extraction are demonstrated; equivalence to field-off full transport is not. The archived reference's
unresolved generator association is a separate limitation, not present in this file's strict pointer validation.

## Persisted selection scope

The source file's embedded `Primary` FHiCL confirms an event-level `FilterStepPointMomentum` on `g4run:crvStage1`
with minimum momentum zero, not a primary-muon selection. Its DS/TS in-volume cut writes a stage step and stops
that track. The inclusive path runs `genCounter` after this selection. Both raw CRV and stage steps are saved.

Reading the retained stage-step product with event labels and joining exact SimParticle keys to the validated
tables finds 122,873 stage steps: 104,215 from primary muons, 18,621 from nonprimaries, and 37 from other primary
PDGs. Of 106,796 retained events, 104,170 have direct primary-muon DS entry and 2,626 do not. Exact parent-tree
resolution shows that 1,932 of those 2,626 events have stage-entering descendants of a primary muon, while 694 have
no primary-muon ancestor among their stage-entering particles. This is event-level selection by any particle,
not a selection on the primary muon. These counts describe the retained file only, not the population of rejected
events or a selection efficiency.

| Group | Primary-muon particle/bar crossings | Same muon has a DS-entry step | No direct DS-entry step for that muon |
|---|---:|---:|---:|
| R1–R6 | 44,049 | 42,023 | 2,026 |
| L1–L3 | 34,635 | 32,241 | 2,394 |
| T1–T5 | 427,192 | 418,598 | 8,594 |
| E1–E2 | 1,559 | 1,425 | 134 |
| U | 1,068 | 883 | 185 |
| D1–D4 | 11,470 | 11,090 | 380 |
| C1–C2 | 1,974 | 1,939 | 35 |

Particle/bar rows remain distinct from V1 event/bar totals: two top-group rows involve another primary sharing a
bar in the same event. No generator-index join or nearest-state association is used. Stock text timing is rounded
to 0.001 ns; earliest CRV entry precedes DS entry for the directly entering primaries at this precision. This does
not test missing post-DS returns, which S1 deliberately does not transport.

Unmatched full-reference and S1 spectra are useful scope diagnostics, not closure tests: the R-group mean path
lengths are about 39.77 versus 24.98 mm, and U about 29.95 versus 20.88 mm. Selection, field, physics-list and
world effects are confounded. The archived full-reference flat tables lack DS-entry StepPointMCs and end-volume
identity, so an exact S1 selection cannot be reconstructed from those tables. Do not substitute a generator ray,
end-state proximity or a dropped subset of unresolved generator identities. `production_scope` records this
unsupported matching explicitly and keeps descriptive spectra for all 23 sectors.

### Reuse disposition

- Reuse the validated production file for **pre-DS, S1-conditioned** particle/bar and response studies, with the
  original exposure and explicit event-level selection. S1 is not a per-primary-muon DS-selection sample.
- Retain the existing full 1M reference for the all-generated, field-off V1 definition. Do not combine it with S1
  as if they sampled the same population or derive a correction from the unmatched mean shifts above.
- Before any more input recalls, check whether already-existing full-reference art retains enough stage/surface
  information to reproduce the S1 event predicate without new transport. Flat CRV tables alone do not.
- If that information is absent, make the target observable explicit and obtain an agreed selection/field/physics
  matching strategy. Any proposed limited continuation or new transport must address a quantified residual need;
  the 19M extension remains on hold. Fixed-stock response seed replication can reuse existing G4 separately.

## Selection and normalization

S1 keeps events passing a DS-entry StepPointMC filter and stops tracks at DS-entry volumes. It saves raw CRV
StepPointMCs, GenParticles and associated SimParticles for those events, not all rejected primaries. Do not
pretend that a selected S1 sample is an all-generated, full-transport reference, or that a missing gen stream
precludes an independent comparison. Post-DS return crossings and CRV-only events require separate coverage checks.

The original log for subrun 517 records 2,500,000 generated events, 106,796 retained events and 29.5618 s of
simulated CRY livetime. The catalog `dh.gencount` is 106,796 because the S1 counter is after filters. It is not the
generated-event denominator. Validate each extracted file against its own original log, catalog metadata and
persisted subrun livetime; a partial-event dump does not represent the full livetime. Do not scale the file count
by a nominal generated count and present it as a verified exposure.

## No-transport extraction

`art/CrvFastSim/fcl/dump_run2_s1.fcl` uses the released `CrvStepsFromStepPointMCs` module on **only**
`cosmicFilter:CRV`, then the existing `CrvStepTableDumper` with the matching `cosmicFilter` SimParticles and
`generate` GenParticles. Pointer identity validation must pass; never repair joins by proximity or time.
The stock conversion needs particle constants, not a geometry service, photon lookup tables, conditions changes
or Geant4. It constructs CrvSteps from existing G4 steps; it does not simulate transport or optical response.
The dump keeps `fullGeneratorStream:false` and an event census. A CosmicLivetime printer records exposure;
`probe_run2_s1_products.fcl` adds a DataProductDump for a **one-event** product probe, not the full extraction.
Read all configured sectors, including sides, U/D and C.

Run only on an existing built area containing that dumper, for example through the documented Offline-on-Slurm
runner, with one previously staged input. No build or tarball creation is required. A zero-event configuration
smoke does not prove product tags or source completeness; the production-file extraction remains a separate gate.

```bash
bash "$OFFLINE_RUNNER" "$OFFLINE_AREA" "$CRV/art/CrvFastSim/fcl/dump_run2_s1.fcl" "$INPUT"
python3 -m crvfastsim.analyses.production_reuse \
  --production-table s1steps.root --metadata source.metadata.json --log original-production.log \
  --output NEW.json
```

The auditor requires the full selected-event census, checked truth associations, matching per-event row counts,
exact source-file/log binding, retained catalog counts and original generated livetime. It reads the highest ROOT
cycle. A successful extraction is **conditional S1 evidence**, not a V1 acceptance certificate.

## Independent-reference precision design

The same module reads the existing paired reference without rerunning either transport arm:

```bash
python3 -m crvfastsim.analyses.production_reuse \
  --reference-root "$REFERENCE_ROOT" --subruns 101 102 103 104 105 106 107 108 109 110 \
  111 112 113 114 115 116 117 118 119 120 --output NEW.json
```

Per generated event, it accumulates `[crossing count, summed length, summed visible energy]`, including zeros,
using the established V1 event/bar aggregation of truth-identified primary muons. Disjoint shards add raw moments.
The event-cluster covariance gives single-arm marginal SEs and hypothetical independent two-arm errors when an
equivalent reference has 1, 10 or 20 times the exposure. Even an infinite reference leaves the fast arm's marginal
variance: removing a paired covariance can cost precision. These estimates are statistical design diagnostics,
not a claim that S1 and full transport have identical selection or a demand for more G4. The fast arm can grow
without new Geant4 if a valid independent production reference and matched normalization are established.

Prefer existing production for conditional/pre-DS studies and retain the current 1M paired reference to diagnose
coverage differences. Only a demonstrated, quantified gap can justify limited continuation or new surface G4.
The fixed-stock response-seed floor can already reuse existing G4 events and is a separate task.

## Actual projection-world difference

Original logs, not just source includes, show the S1 production world has x ±36,786.6 mm and z ±52,264.6 mm,
whereas full-reference shard 102 has x ±531,787 mm and z ±547,265 mm. Both use y −3530.2…15365.4 mm, the same
CRY reference point and the same target box. The later reference adds 500 m cosmic-world margins. Compare
projection entry surfaces and outer-material coverage, especially for grazing cosmics; equal top-level
`geom_run2.txt` hashes do not remove this difference. This is a comparison axis, not a reason by itself to discard
existing production. `--reference-worker-log` records all three actual projection settings alongside their hashes.
Do not subtract differently conditioned S1/full-reference acceptance fractions to infer a coverage correction.

The optional projection diagnostic tags a generated muon's **straight ray** by whether it enters the smaller
production world through a side rather than the top, then counts that tag among G4 primary-muon/bar crossings.
It is a truth diagnostic, not a measured-variable correction, transport trajectory, DS-entry selection or efficiency.
Retained G4 generator indices may be compressed: use SimParticle→retained-gen identity first, then an unambiguous
**exact stored-value** retained-gen→full-gen subset map. Reject missing or duplicate-value matches; never join the
compressed index directly to the full stream or choose a nearest momentum/position/time match. The mapping is
for this archived exact-subset fixture, not a general fuzzy ancestry algorithm.
