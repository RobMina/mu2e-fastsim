# Aggregated board accessor extraction

`beam/art/FlashBoard` is a small read-only art satellite. Its
`FlashBoardExporter` reads **existing** StrawGasStep, CaloShowerStep and CrvStep
collections plus SubRun GenEventCount. It does not run Geant4, reproduce raw-step
aggregation, apply a detector filter, edit event products, or generate a fast
prediction. It is separate from the frozen R5 raw exporter and deployments.

**Build/runtime status:** Python checks pass, and both compiled plugin libraries
were observed after the corrected user build, with timestamps newer than their
sources. The build log, runtime/plugin identity, synthetic art fixture and
real-file extraction remain unverified. A library file or successful Python test
does not establish ABI compatibility, FHiCL expansion, art scheduling, or actual
pointer resolution.

## Physics purpose

The [historical detector selector](flash-board-selection.md) needs exact
aggregated accessor values, original collection order, product-qualified particle
identities and explicit empty products. Raw StepPointMC tables and the existing
tracker-only dumper do not supply this contract. The satellite fills that
extraction gap without changing the simulated sample.

The initial target is reading historical selected anchor art with Offline
v13_39_00/p107, the backing used by the original analysis. It reads persisted
aggregated products; it does not regenerate them with v13_39 defaults. The
historical production aggregation was v13_38_00. A future raw-to-aggregate
comparison on the frozen inclusive v13_41_01/p109 sample needs its own matching
runtime/configuration review and is not implied by this reader.

## Minimal output contract

The analyzer label sets the ROOT directory, conventionally `board/`. The exact
v1 column contract is in `beamfastsim.flash_board_export.SCHEMA` and the
[satellite README](../art/FlashBoard/README.md).

- `schema`: one version and configured expected-event/SubRun-count row.
- `requirements`: every configured detector collection, indexed by detector
  role and configuration slot. Roles: tracker=1, calo=2, CRV=3.
- `events`: one row per fully exported event, including empty events, with
  product and per-detector row counts.
- `products`: every required collection including size-zero products, its
  actual ProductID, configured slot/tag, and resolved branch/module/instance/process.
- `straw_steps`, `calo_steps`, `crv_steps`: exact stored energy/momentum
  accessors, original collection indices and particle ProductID/key. No step
  selection, PDG restriction, time cut or photon weight is applied.
- `subruns`: exact generated counter with independent SubRun identity and
  product provenance, including SubRuns with no selected tracker response.
- `completion`: endJob counts after the configured expectations have been met.

Event keys are uint32; ProductID, pointer keys, indices and counts are uint64.
Accessor columns are float32. Compile-time assertions require the historical
accessor return types; the exporter calls `momentum().R()` / `startMom().R()`
directly rather than reconstructing magnitudes from float64 components. Missing
collections or null/unavailable particle pointers throw; missing is never empty.
Repeated EventIDs, repeated SubRun callbacks/fragments, aliased configured
products, invalid accessors and counter sentinels fail explicitly. It does not
sum repeated counters or guess which duplicate should win.

No source ancestry table is required for this selection-only export: each step
pointer must resolve, and its exact target product/key is exported. This is not
an ancestry or inclusive photon-response replacement.

## Python reader and scope

```sh
python3 -m beamfastsim.flash_board_export \
  --input EXPORTED.root --out CHECK.json --events N --subruns M \
  --input-scope retained
```

The reader requires native TTrees and reads their latest cycle. It preserves
integer identities and stored floating precision, checks schema/row/count
consistency, verifies every configured slot in every event (including additional
collections of the same detector), checks declared product sizes against original
indices, then uses the historical detector selector and observable reducer.
It writes JSON exclusively: existing output, including the input path or aliases,
is never overwritten. `--events` and `--subruns` are independent expected counts,
not read limits. A completion tree is necessary but not sufficient: successful
job termination and a closed readable file still have to be checked externally.

Two input scopes are deliberately distinct:

- **`retained`**: the exported input is the historical selected stream. Every
  supplied tracker deposit contributes to the saved observable, even if the
  selector's diagnostic replay disagrees. Disagreement returns a nonzero CLI
  status and is reported; it must not be hidden by deleting those events.
  Refiltering selected files cannot validate events absent from the files.
- **`filter-input`**: the exported ledger covers the events actually presented
  to the detector filter, including rejections. Only events accepted by the
  replay contribute to the selected quantity. This does not certify earlier
  G4/status/prescale gates.

**Normalization is withheld by default.** An art `-n` read limit can expose a
full SubRun GenEventCount while processing only part of its selected events.
Count/marker agreement alone cannot catch that error. Only append
`--complete-subruns` after establishing that the input inventory covers every
selected event for every counted SubRun, including all zero-retained SubRuns.
Otherwise `mev_per_generated_event`, `retained_fraction`, and the central budget
decision are `null`; exported and selected energy sums remain available.
The explicit declaration is a caller obligation, not an automatic census or
new proof of exposure completeness. Never pass it merely because an arbitrary
small-N read finished successfully.

The documented budget remains 7.506758e-6 MeV/generated resampler event. No
anchor-fitted factor, POT conversion or inferred uncertainty is introduced.
Sequential source reuse, upstream failures and all-species response remain
separate validity requirements.

## Bounded validation sequence after the user build

1. Run the four-event/two-SubRun synthetic fixture in a new data directory.
   Require successful job completion and validate its exact table payload,
   selection masks and independent counters. The second SubRun has no accepted
   events but still contributes its counter.
2. Re-read the fixture's saved art without any producer and compare every typed
   payload, not just the final energy sum. Exercise missing-product and invalid
   pointer configurations in separate outputs; neither may produce a passing
   completion result.
3. Inspect one historical anchor input's product census, then export its existing
   products with exact tags and independently known event/SubRun counts. For a
   partial read, keep normalization unavailable. For the complete historical
   inventory, compare retained event counts and tracker totals to the original
   receipts: bpzup21 14,407 events/40.74233601982624 MeV; bpzup12 53,729 events/
   149.9907398049316 MeV; each 11,000,000 generated events in 100 SubRuns.
4. Only after extraction closes, prepare historical aggregation on a bounded
   frozen inclusive art sample without rerunning Geant4. Preserve raw files,
   use matching saved geometry/conditions, and explicitly configure delta
   combining and calorimeter/CRV aggregation. Missing source resampling/pulse
   and changed transport runtime remain limitations, not normalization factors.

No new simulation or request to rebuild Offline itself is part of this change.
The user builds the satellite on the designated build host; execution waits for
that build to finish and be verified. The prepared area's `backing` link was
checked remotely to resolve to Offline v13_39_00; its p107 build directory and
Offline envset declaration were inspected. `prepare_board_area.sh` checks the
repository marker/SConscript and backing layout/envset, then creates a
non-replacing link; it is not a release-identity or binary-freshness certificate.

The first user build was a no-op: `beam/art/.muse` pinned p107, but the missing
`FlashBoard/.muse` meant Muse did not register the satellite. The repository
marker is now supplied. Start a fresh shell so `muse setup` refreshes discovery;
`MUSE_LOCAL_REPOS` must contain `FlashBoard`. No object or plugin files were
produced by that first attempt, only `.musebuild` timestamps.

On the designated build host, after setting `BOARD_AREA` to this checkout's
`beam/art` directory and initializing the standard Mu2e environment:

```sh
cd "$BOARD_AREA"
setup muse
muse setup
printf 'MUSE_WORK_DIR=%s\nMUSE_LOCAL_REPOS=%s\n' "$MUSE_WORK_DIR" "$MUSE_LOCAL_REPOS"
[[ "$MUSE_WORK_DIR" == "$PWD" && " $MUSE_LOCAL_REPOS " == *" FlashBoard "* ]] && muse build -j 32
```

This builds the two satellite modules against the existing backing, not a new
Offline checkout. No build command is embedded in the preparation script. Source
and runtime checks precede any fixture or historical-data execution.
