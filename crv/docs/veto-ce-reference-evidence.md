# CE reference discovery and retained-worker provenance

## Decision

Read-only documentary discovery advances **one upstream digitization execution
binding**, but does not establish an independent CE calibration/validation
reference. No physics payload, fit, response draw, stock rerun or new split was
used. Reuse the completed availability, keyed rows, joint summary and composition
reports. The requirements in
[veto-ce-calibration-validation.md](veto-ce-calibration-validation.md) remain in
force: C4 stays stock, historical F is unchanged, and finite-library uncertainty
is unestimated rather than zero.

## Scoped catalog discovery

Seven initial SAM queries completed successfully: four `list-definitions`
filters (`*FlatPBI*`, `*Luminosity*`, `*Intensity*`, `*CeMLeadingLogMix1BB*`) gave
empty replies; the exact digitization worker-zero log-family count was one and
two hypothesized reco/ntuple worker-zero families returned zero. These are
**query-specific results, not evidence of absence**. In particular, a follow-up
exact `describe-definition` and exact `list-definitions --defname` both find
`dig.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.art`, definition ID 220224,
created 2026-07-26T17:46:30+00:00. The earlier wildcard filters therefore cannot be
used to exclude this known family, or any independent reference. No broad
catalog scan was attempted, and the filter behavior was not diagnosed further.

The follow-up enumerations use `with limit 2` and require exactly one matching
worker log. Queries are read-only, individually capped at 256 KiB and bounded by
a 60-second process timeout plus SAM socket timeout. The initial discovery used
SAM timeouts without the process wrapper. No query was retried or successful
capture overwritten.

## Worker index is not output sequencer

The initially discovered log is:

```text
log.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.0-1785087048.log
SAM file ID: 106906948
Bytes: 449780
SHA-256: dc2b977ec44bf477c0aa68f7f124ca863b9fb29eac41f53b14ee843c37b35143
```

Its exact bytes agree with the catalog digest. The runtime record says global
and local worker index **0**, but writes
`dig.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.001430_00000033.art`, using CE
dts inputs 33, 49, 108 and 91. This is not retained output zero and cannot be
substituted for its worker evidence. This discrepancy is a discovery-assumption
failure, not evidence of a production defect.

The already captured, hash-checked job-definition JSON groups four CE source
files per worker. Entries 48–51, zero-based, are subruns **150, 0, 185, 136**:
group **12**. These agree with the earlier retained-output-zero catalog parents
and provide a worker-12 discovery hypothesis. Only the following runtime record
establishes its actual output mapping.

## Retained output zero: verified digitization worker

```text
log.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.12-1785089806.log
SAM file ID: 106907717
Catalog creation: 2026-07-26T18:16:48+00:00
Bytes: 449984
SHA-256: 735043e5ec4ba594263a49e161e5ebb2ff043c18eeeb82bb425987200f14227b
```

The exact known catalog path is under
`/pnfs/mu2e/persistent/datasets/phy-etc/log/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_3/log/0f/57/`.
The read was capped at catalog size plus one byte, timed out after 60 seconds if
necessary, and required exact size and SHA-256 agreement. No namespace listing,
stat, payload staging or physics-file open occurred.

Evidence in this log, using its original line numbers:

| Evidence | Lines / observation |
|---|---|
| Worker and definition | 27–32: PROCESS 12, global/local index 12, definition archive index 0 |
| Generated FCL | 35–95: Mix + OneBB includes, Sim_best/v1_3, PBISim SDF 0.8, baseSeed 13 |
| Actual CE source list | 49–54: subruns 150, 0, 185, 136, in that order |
| Actual output | 92, 125, 241: generated name, opened and closed output `001430_00000000.art` |
| Runtime environment | 98–112: SimJob/MDC2025au, al9-prof-e29-p103, art 3.15.00, ROOT 6.32.06, KinKal 03.07.00; loaded GlobalConstants from Offline v13_33_00 |
| Processed input events | 600: art event summary 11058 |
| Written output events | 2643 and 2848: Output count 8212 |
| Execution status | 2933–2934: art status 0 and successful Mu2e completion |
| Output handoff | 2935–2937: exactly the retained output name selected for pushOutput |

The generated background input lists name MuBeamFlashCat 77, NeutralsFlashCat
579, MuStopPileupCat 23 and 13, plus 25 EleBeamFlashCat inputs retained verbatim
in the log. Runtime warnings explicitly record wrapping to initial mixing files,
including NeutralsFlashMixer at run1430/subrun150/event1134 after 90878 secondary
events and again at run1430/subrun136/event4494 after 1543751. These are observed
reuse in this worker, not independent-history counts or a new covariance estimate.

The written count agrees with the **previously saved** metadata for dig output
zero: SAM ID 106908564, 8212 events, 4985071320 bytes, declared SHA-256
`854cdd052613a682fd7624b6222e5d1062857645088969da8c5fa3a9e66f068b`.
Its saved mcs descendant record likewise names that dig parent and 8212 events.
No fresh physics-payload hash was computed. The 11058 art input count and catalog
`gencount=20000` are different quantities; neither is promoted to an audited
generated-CE denominator or additive POT here.

This proves a documentary mapping for **one digitization worker**, with concrete
runtime environment and input/output observations. It does not certify all eleven
retained sources, downstream reco/ntuple workers, selection imports, track-fit time
conventions, all conditions/IOVs, full resolved FCL or original physical ancestry.
The log records an RCDS prodtools tarball path, not a verified release version or
binary hash; no deployment tag is inferred from a different campaign's worker.
The earlier recipe/source semantics remain separately documented in
[veto-ce-proxy-semantics.md](veto-ce-proxy-semantics.md).

## Independent-reference candidates: none admitted

The following are **available local source/document snapshots**, not authenticated
measurement records or proof of their historical deployment. The Offline and
Production source mirror was paused; its contents were inspected, not executed.
No IHEP data file, STM coefficients/database, KPP event data or new simulation
sample was read.

| Lead | What it establishes | Missing before admission |
|---|---|---|
| `Offline/EventGenerator/test/FitBeamIntensity.C`, `FitPBI.C` | Macros read `IHEP_PBI.dat` as relative intensity/frequency, say some bins were not measured, and assign every bin error 0.02; the second labels the comparison MDC2018 Generator | Measurement/publication owner, instrument/run, calibration, sampling and absolute normalization, justified uncertainties, transport to target conditions, event-linked joint intensity/phase/selected-time law. The hardcoded errors are not independently validated measurement errors. |
| `Offline/TrkHitReco/src/ProtonBunchTimeFromStrawDigis_module.cc` and `TrkHitReco/fcl/prolog.fcl` | Detector-input timing procedure: calibrated TDCs, earlier straw end below 200 ns, minimum 100 hits, mean minus 99.5 ns; low-hit fallback is 12.5 ns with 25/sqrt(12) ns error, despite a stale zero-value comment. The offset has a database FIXME. The prolog defines PBTFSD but makeSH points to EWMProducer. | Actual activation, measured rather than MC input, calibrated offset/IOV and beam-clock reference, selection and shared-tracker correlations. Source existence is not a deployed reference. |
| STM testbeam source, reconstruction and STMEnergyCalibMaker | HPGe/LaBr energy-calibration infrastructure; testbeam sampling differs from experiment and configuration references STMAnalysis text files | Stopped-muon-rate normalization, efficiency/exposure and event/time linkage to the CE population. Energy calibration alone does not supply these. |
| KPP run124155 tracker–CRV timing documentation | Real-data cosmic relative timing offset +495 ns | Absolute beam clock/intensity reference and justified transport across geometry, conditions and cosmic/CE domains. |
| `Production/JobConfig/mixing/FlatPBI.fcl` | A Run1 uniform-PBI study recipe: mean1.18e8, halfWidth1.0, nominal range0..2.36e8 | Registered executed sample, worker/inclusion-law binding, target distribution, overlap and uncertainty propagation. A configurable simulation recipe is not independent calibration or permission to weight existing events. |

Snapshot SHA-256 values (path labels above):

```text
FitBeamIntensity.C 3ded38bd63399556759e8402590d26087cb0658e8f52b1b097d7ba838e56c735
FitPBI.C 3a5313725f958d7614ab335629945843cc0c7b05ebc8b9bb07f8eb421fcdc1a7
ProtonBunchTimeFromStrawDigis_module.cc 7791b359f2dc3f9cecfc5cf90aaf2b8fe098f9ed4139ebe0ed66625fee6be1cc
TrkHitReco/fcl/prolog.fcl c5f01f6ecc2b5e0f1c0dfbd86405d12c38795784aab88276cc315f8bf62efb40
FlatPBI.fcl bd7da65cd07718edad93abefbed35d2b8f48ec20b86dc503964237abc6793f7e
Sources/test/FromSTMTestBeamData.fcl b0d37cc49a60ca9cd5eab02831ce928a1e12b62a0e7c460dccd943e9a61d0cdf
STMReco/fcl/prolog_testbeam.fcl 8c6aeb464b3e43a7f76eb5481f6696b70d7aee2e50c87de5b334805dd3f0f43c
STMReco/fcl/makeSTMHits_testbeam.fcl 7a488fce177c05dd44cb6d69552cd26bb38d5aa64700e3a358616e3444f325ec
STMConditions/src/STMEnergyCalibMaker.cc 51b3a01dd2dfa21fcdf6f50a30f81d4dbae8f2a2572683604e6f883bece58868
mu2e_track_crv_mcresponse/README.md 17bee9080568e5515d9a43c9874d7853dd9d49499f7244d5f791aa5c1c03f82c
```

The live service registry identifies the token-protected KB endpoint. A bounded
unauthenticated request returned HTTP401 with `invalid_token` / `Authentication
required`. **No authenticated KB search was performed**. This is a service-access
limitation, not evidence that the required literature does not exist. No credential
cache was inspected, copied or refreshed, and no retired local document service
was used. An approved KB connection or a supplied measurement/reference is needed
to pursue that lead.

## Captures and remaining gate

The new documentary scripts are `scripts/discover_veto_ce_reference.sh`,
`capture_veto_ce_worker_catalog.sh`, `capture_veto_ce_worker_log.sh` and
`capture_veto_ce_retained_worker.sh`. Shell syntax checks pass; each real capture
completed once, with exact mirrored script-hash checks, new output directories,
byte caps and checksum manifests. This is not an analysis test suite or a
calibration validation. Measured capture costs:

| Capture | Wall seconds | Peak RSS KiB |
|---|---:|---:|
| Initial seven catalog queries | 0.93 | 23224 |
| Worker-zero catalog and exact definition | 0.71 | 23204 |
| Worker-zero documentary read | 0.84 | 15872 |
| Saved group12, worker12 catalog and documentary read | 3.74 | 23144 |

No capture needs rerunning. The next calibration gate remains a named independent
reference with target/inclusion law, measured features, time/conditions provenance,
and genuinely unexamined ancestry-separated validation. In parallel, historical
execution coverage still needs the remaining digitization and downstream workers,
and the exact selection/fit-time deployment. Do not substitute filename indices,
catalog checksums, a marginal relative-PBI spectrum, simulation truth labels, or
retrospective splits of the eleven viewed files for those requirements. Broader
physical-library covariance and full photon-arrival/step-time closure remain open.
