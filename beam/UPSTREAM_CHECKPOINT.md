# Upstream source checkpoint and resumption guide

Development paused on 2026-11-05. This branch is a **work-in-progress archive, not a promotion or merge-ready transport implementation**. Changes are opt-in upstream additions; the existing DS/CE execution defaults are not changed by the upstream modules. The branch retains its original umbrella base and is not rebased onto newer main. Reconcile subsequent main/submodule changes separately before integration; do not interpret an archive diff against a newer main as a reviewed integration patch.

## What exists

| Area | Delivered source/evidence scope | Not delivered |
|---|---|---|
| Existing-data search | Read-only catalogue tools, retained-row and G4Beamline extraction, configuration/selection audit contracts | Verified unbiased inclusive target-exit library; exhaustive absence proof |
| Field/geometry/numerics | Restricted map readers/selectors, placed-solid and boundary diagnostics, conditional vacuum/decay/uncertainty kernels | Qualified native geometry, practical full-path material-aware PS/TS transport or accepted physics accuracy |
| Native recorder | Unapplied patch, default-off helper and exact base/postimage/asset pins; actual-source, primary-link, insertion/omission and assembled in-memory reconciliation draft | Compilation/linking, runtime callback/geometry qualification, actual recorder-produced ledger, launch frontier |
| Original-product reader | Standalone uninstalled EDAnalyzer source, separate pins, bounded structural readback join | Actual original-art readback, authenticated file/registry/process/config/output selection, successful producer/reader job and shard/weight/POT closure |
| Validation/performance | Recorded scoped software controls only | Paired held-out Run 1A TS-exit/DS-entrance/full-chain closure, device transport, matched throughput or physical event-cost measurement |

The latest recorded pre-pause source-only check passed **852 focused controls** and **1,910 scoped Python tests plus 158 subtests**. The focused controls overlap the scoped suite; do not sum them as independent coverage. Exact recorder assembly checked **25 base files and 12 patched postimages**, not a complete Offline/muse area. The separate reader was not installed, compiled or executed. Full inherited CPU/JAX regression remains incomplete. No test duration is a transport benchmark.

The final documentation audit corrects stale summaries, missing links, provenance/cap descriptions and a rigid-transform qualification; documentation/hash bindings were refreshed deliberately. Test docstrings were cleaned, and the source-check wrapper now requires site paths through environment variables. Those pause-time edits were not given a new pytest run; the recorded pass belongs to the preserved pre-pause snapshot, not to an invented final execution receipt. Native C++/patch and Python algorithm bytes remain unchanged by the pause audit.

## Documentation map

- [UPSTREAM.md](UPSTREAM.md): detailed contracts and limitations of numerical/reference prerequisites.
- [TARGET_LIBRARY.md](TARGET_LIBRARY.md): inclusive source, zeros/companions, exposure, re-entry and validation acceptance requirements.
- [TARGET_RECORDING.md](TARGET_RECORDING.md): native observation, geometry, lifecycle, causal-frontier and approval requirements.
- [TARGET_SOURCE_SCHEMA.md](TARGET_SOURCE_SCHEMA.md): v2 input/link/insertion/in-memory rows; v1 remains source-unverified.
- [TARGET_SOURCE_ADAPTER.md](TARGET_SOURCE_ADAPTER.md): source/primary/product boundaries and reviewed callback limitations.
- [TARGET_READBACK.md](TARGET_READBACK.md): separate original-product reader schema, file/SubRun closure, structural join and remaining authentication gates.
- [native/README.md](native/README.md): exact recorder subset assembly; not a build recipe.

`native/source_pins.json` binds the recording helper/patch/README and exact upstream base/postimages. `native/readback_pins.json` separately binds the reader, join and readback documentation. A recording assembly receipt does **not** compile, install or authenticate the reader. Updated documentation hashes do not create new native evidence.

## Resume in this order

1. Read the acceptance/recording/readback contracts and inspect the preserved checkpoint diff. Keep the library-first priority; do not resume by extending numerical infrastructure instead.
2. Complete original-file/hash/registry/process/config/output-selection binding and independently checked request/source/native/written EventID, SubRun, job, shard and weight accounting. A filename, caller request, nonempty provenance string or normal status is not authentication. Keep completed-proton count and POT normalization unset until established; missing events are not zeros.
3. Implement the causal launch-frontier graph using continuation and production-step incidence. Earlier-born daughters remain independent; transfer owns later continuation/branches/returns. Refuse unresolved same-step birth/transfer order. Qualify target membership and upstream/exterior ownership separately.
4. Review the complete proposal for build readiness before preparing a full isolated current-source area. The user performs builds. Compilation, no-transport original-art controls, native inventory/probes and one-event-first Geant4 diagnostics are distinct approvals, with explicit event/site/wall/CPU/output/recovery limits.
5. Only successful inclusive/persisted controls can justify a separately approved production library and held-out physics validation. Retain target material dynamics for re-entry; do not replay stock returns as extra launches. Complete material-aware PS/TS physics before device/throughput work.

Existing parser-only `fcl/target_library_diagnostic_run1a.fcl` is **not ready for native library generation**. Do not submit its 100-event placeholder as an approved campaign.

## Source-only checks when resuming

The existing analysis image and compatible pinned core must already be available. No installation, build or transport is embedded in the wrapper. In an approved single-CPU Slurm allocation, set:

```bash
export FASTSIM_CORE_ROOT=/path/to/pinned/core
export FASTSIM_IMAGE=/path/to/existing-analysis.sif
export FASTSIM_BIND_ROOT=/path/to/shared/input-output-ancestor
bash beam/art/tsrecordproposal_checks.sh /path/to/source-snapshot /path/to/new-output-root /path/to/pinned-Offline-subset
```

The source snapshot contains `beam/`; the bind root must cover the snapshot, image-accessible inputs, pinned core, source subset and output paths. Output assembly/report paths must be new. This checks source hashes, synthetic ledgers/readback and scoped Python tests only. Do not treat it as permission to compile C++, construct geometry or run Geant4. Site logs, raw files, job identifiers, accounting and immutable execution receipts remain outside this public source archive.
