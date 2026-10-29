# Historical production configuration evidence

This documentary follow-up identifies cataloged job definitions for the retained
Neutrals and CE datasets. It does not run simulation, reconstruction, metadata
extraction on physics files, or an uncertainty calculation. All previous saved
controls and reports remain unchanged.

## Evidence levels

1. **Catalog observations:** exact-file SAM responses, with output identity,
   parents and declared counts. A catalog hash is not a fresh physics-file hash.
2. **Registered recipe bytes:** three job-definition tarballs read at their exact
   catalog locations and SHA-256 checked against SAM. Small FCL/JSON members are
   retained separately. This establishes their contents, not execution by every
   output file carrying the campaign name.
3. **Release-source interpretation:** sources at the setup paths named in those
   recipes, copied with before/copy/after hashes. These are selected sources, not
   a complete transitive configuration expansion or deployed-binary certificate.
4. **File-specific worker evidence:** the Flash 13147 follow-up below binds its
   generated FCL and prodtools version and identifies a parent-list defect. This
   does not establish execution or metadata-declaration packages for every file.

## Registered job definitions

| Role | Catalog filename | SHA-256 |
|---|---|---|
| Flash resampling | `cnf.mu2e.NeutralsFlash.MDC2025ac.0.tar` | `8785cba6fcbd6590ef8ec58035c9cd15966cb7bfb0849cf344bb9086dc1b0e9a` |
| Flash concatenation | `cnf.mu2e.NeutralsFlashCat.MDC2025ad.0.tar` | `c17855fe607c8233b160e1158767bee5b4f9c80241da1196201297c80b3651fb` |
| CE ntuples | `cnf.mu2e.evnt.MDC2025au_best_v1_1-001.0.tar` | `863ae9650e84613a78d9e2b82b8fb15864ae74df89f628dd1c51e749d4728f18` |

Their catalog creation dates are respectively 2025-09-18, 2025-11-13 and
2026-09-03. The two small tarballs contain complete captured `jobpars.json` and
`mu2e.fcl` members. The Cat archive is **18,068,743 bytes compressed**, with a
**360,000,549-byte `jobpars.json` member**. The initial capture retained only a
bounded prefix and tail, plus the 122-byte Cat FCL. The subsequent bounded audit
streams the complete JSON without materializing its list: **5,000,000 filename
entries**, with all twelve retained 5,000-name blocks matching saved SAM parent
hashes. Its full inventory includes two timestamp-only PAX headers. See
`veto-cat-jobdef-audit.md` for bounds, pins and limitations. No `jobquery` or
current FCL generator was run on this large archive.

## Neutrals resampler recipe

The registered Flash recipe names
`/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac/setup.sh`;
its inspected backing resolves to `Musings/Offline/v10_47_00`.

- `source.firstRun = 1430`, `source.maxEvents = 1340000`.
- Subrun key: `source.firstSubRun`; seed key: `services.SeedService.baseSeed`.
- Forty explicit `sim.mu2e.NeutralsCat.MDC2025ab.*.art` auxiliary filenames,
  one requested per job, `sequential_aux = true`.
- Embedded FCL includes `Production/JobConfig/pileup/NeutralsResampler.fcl`,
  sets Run 1A geometry, `bfgeom_no_tsu_ps_v01.txt`, and
  `physics.filters.neutralsResampler.mu2e.MaxEventsToSkip = 1335651`.
- The named release's pileup prolog uses sequential reading, `wrapFiles: true`,
  and `[genCounter, protonTimeOffset, neutralsResampler]` before the filters.
  Its resampler requests one secondary and selects the first skip uniformly
  over the inclusive integer range zero through `MaxEventsToSkip`.
- The output retains `GenEventCount`; that release's counter counts producer
  executions. This is a resampling-stage count, not an ancestor-unique POT count.

The Cat recipe names `SimJob/MDC2025ad`, whose inspected backing resolves to
`Offline/v11_00_00`. Its saved FCL includes `common/artcat.fcl` and an output-name
pattern. That release's artcat configuration uses RootInput/RootOutput, no new
counter, and `fastCloning: false`. The subsequent full streaming audit confirms
`source.maxEvents = 2147483647`, 5,000 requested source filenames and
`sequencer_from_index: true`, including the complete surrounding JSON schema.
All twelve retained declared blocks agree with catalog parent sets; this is not
an execution receipt or an audit of the other 988 blocks.

### Catalog ancestry and an assignment counterexample

Two immediate parents from the already saved Cat-744 parent list were queried:

| Flash subrun | Saved catalog parent NeutralsCat subrun | `dh.gencount` | Event count |
|---|---:|---:|---:|
| 13147 | 32205 | 1,340,000 | 22 |
| 36317 | 34446 | 1,340,000 | 20 |

The queried upstream `sim.mu2e.NeutralsCat.MDC2025ab.001430_00032205.art`
record declares `dh.gencount = 2500000`, with 1,333,172 events. These observations
extend two ancestry edges; they do not census all retained parents, identify
independent showers, or settle cross-species sharing.

**Do not infer assignment as `subrun % 40`.** The inspected prodtools v1.3.5
implementation would use that rule for one sequential auxiliary file if the
subrun equals its job index. It predicts list index 27 / parent 32644 for Flash
13147, whereas SAM records parent 32205 at list index 28. For Flash 36317 it
predicts index 37 / parent 34446, which agrees. Thus **one of two checks disagrees**;
there is no basis for a universal offset repair. At that stage, actual job-index
and worker FCL evidence were missing. The file-specific follow-up below now
resolves 13147 as a parent-list defect; it does not certify every worker's mapping.
No assignment was silently changed.

`5000 * 1340000 = 6700000000` is an arithmetic consistency observation. Only two
Flash counts were queried; neither this product nor the Cat catalog count proves
that every parent contributes that count, that any files are independent, or that
physical exposure is additive. No library weights or covariance were calculated.

## Metadata extraction rule located

`artexternals/mu2efilename/v3_13/bin/printJson.sh` supplies the previously missing
SAM field-writing rule (source SHA-256
`9b5f211110d85d8ce809fb984bbd3f3a5a23f78e02b632d96eed3c84812e0e65`).
For art files it constructs a `GenEventCountReader` job in `RunsAndSubRuns` mode,
extracts token three of `GenEventCount total:` and writes `dh.gencount`. A missing
count product can omit the field; other extraction errors fail. The reader sums
one count product per subrun and does not deduplicate ancestor reuse.

The inspected `mu2egrid/v8_03_02` worker wrapper calls
`printJson.sh --parents parents` and permits the mu2efilename version to be chosen
by submission settings. Therefore this is a concrete available historical
implementation, **not proof of the exact package deployed for each declaration**.
No extractor was executed on an art file. The modern mdh extractor instead maps
the reader output to `gen.count`; it must not be substituted for the historical
SAM writer without a deployment record.

## CE ntuple recipe and measured-observable limits

The registered CE job definition specifies:

- `AnalysisMDC2025/v02_01_00/setup.sh` (inspected backing: `SimJob/MDC2025av`);
- `EventNtuple/fcl/from_mcs-mockdata.fcl`;
- `source.maxEvents = 2147483647`, empty subrun key, generic output description
  substituted from the input, and no embedded input list.

The exact retained ntuple `...001430_00000000.root` catalog record (file ID
107470730) names `mcs.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.001430_00000000.art`
as its parent. That agrees with the recipe family but is not a worker-execution
receipt. The inspected recipe's release writes version **6.13.2**:

- `EventInfo.nprotons` defaults/resets to zero; no `_einfo.nprotons` assignment
  was found in that release's source.
- `PBITag: PBISim` fills **MC** `evtinfomc.nprotons`; measured bunch time comes
  from `EWMProducer`.
- `lumistream.fill: true` writes detector observables: proton time-cluster count
  from `TrigFlagPH`, tracker-hit count from `TrigMakeSH`, and calorimeter/CAPHRI
  quantities from `CaloHitMakerFast`. Missing handles leave reset sentinels
  (counts `-1`), so branch presence alone does not demonstrate usable values.
- The mockdata FCL disables generated-event-count and cosmic-livetime subrun
  extraction. No all-generated-CE denominator follows from the retained ntuple.

This strengthens the unfilled-measured-PBI explanation for the **declared recipe**.
It does not yet certify the writer executed on all eleven retained ntuples, nor
establish availability, calibration, selection dependence or sufficiency of the
luminosity proxies. No ntuple, version histogram, luminosity rows or selected
tracks were opened. The inherited zero-field observation is not newly measured.
MC intensity remains a truth diagnostic, not a measured correction variable.

## Capture and next checks

`scripts/capture_veto_historical_config.sh` archives small text members, exact
catalog replies and selected release sources. It uses a new output directory,
checks the three tarball hashes before and after, bounds the small extracted
members and source snapshots, and records original paths and hashes. It never
extracts executable archive paths or runs a campaign. Its marker is a documentary
completion note, **not** atomic publication or a production certificate.

Completed capture: 2.90 s wall, 16,256 KiB peak RSS. Archive pathname size:
331,159 bytes. `bash -n`, complete archive-manifest verification and input/source
hash checks pass; no new unit-test or physics-test claim is made.

Follow-up status and remaining minimum evidence:

1. **Complete:** bounded Cat streaming audit, all twelve declared blocks matching
   saved parent hashes. No physics extraction or block-independence assumption.
2. **Resolved for Flash 13147:** exact worker FCL and prodtools v1.3.7 explain the
   catalog-parent defect. Other workers, recovery mapping and exact count-writer
   package deployment are not certified by this one record.
3. **Subsequently completed:** the separately authorized bounded retained-file
   availability audit finds version 6.13.2 and scalar luminosity fields in all
   eleven files; measured nprotons is zero throughout. See `veto-ce-proxy-audit.md`.
   The upstream digitization recipe/source follow-up in `veto-ce-proxy-semantics.md`
   distinguishes raw detector activity and MC-smeared reconstructed bunch time
   from independently calibrated beam intensity. Historical execution/calibration
   remain separate from these observations.
4. **Open:** deeper/cross-species physical ancestry and intended reuse law before
   assigning finite-library uncertainty.

Historical F is unchanged. C4 remains stock; replay-v2 consumers remain fail-closed.

### Follow-up: Flash13147 worker evidence

The exact worker log has now been recovered and hash-verified. Its generated FCL
uses 32644, while the catalog parent is 32205. In the worker's named prodtools
v1.3.7 release, the FCL generator honors sequential auxiliary selection but the
parent-list helper samples instead; the read-only helper reproduces 32205 exactly.
See `veto-flash-parent-metadata.md`. This resolves the observed discrepancy as a
parent-metadata defect for this case, not an off-by-one assignment. The 32205
upstream metadata above belongs to the catalog path, not the runtime FCL ancestry.
