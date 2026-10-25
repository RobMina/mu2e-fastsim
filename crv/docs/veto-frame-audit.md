# Saved-frame deadtime audit

`python3 -m crvfastsim.veto.frameaudit` audits existing mixed-frame outputs. It does not
run detector response, alter cards or selections, or change the `veto_c4` objective.
Physics results and site-specific paths belong in the internal veto wiki, not this repository.

## Usage

From the umbrella, with `core:crv` on `PYTHONPATH`:

```bash
python3 -m crvfastsim.veto.frameaudit "$FRAMES"/frames_mix_*.parquet \
  --selection "$SELECTION" --reference "$REFERENCE" --out "$NEW_REPORT"
```

- `REFERENCE` is the saved `deadtime_stats.json` for this campaign. Its card (hash
  checked), momentum cuts, selected-track count and grid spacing define the audit.
- Each frame parquet must have a same-basename JSON sidecar from `veto.mixframes`.
- Each file is one job/subset block, with a distinct seed. Repeated event IDs across
  jobs are expected; duplicate `(run, subrun, event)` within a job are rejected.
- Frame/event counts, coincidence counts, occupancy, finite coincidence quantities
  and pool denominator bounds are checked against the sidecar.
- Frames with no coincidence must pass the gate and contribute zero coverage, but
  remain in the denominator. This does not distinguish stepless frames from frames
  whose steps produced no coincidence.
- The output records per-job moments, pool sizes, mean intensity/multiplicities,
  timings, sidecar hashes and the reference hash. An existing report is never overwritten.
- The selection and frame payloads themselves are not hashed; this is not a complete
  immutable provenance manifest. The selected-track count check is not event-list identity.

## Uncertainty contract

For frame coverage `c`, the pooled result and IID error reproduce the existing
`deadtime.factorised` calculation on the same grid. The additional job-block error is

```
mu = sum(S_g) / sum(n_g)
SE_block = sqrt(G/(G-1) * sum((S_g - n_g*mu)^2)) / sum(n_g)
```

where `S_g` is the sum of coverages in job `g`. Equal job sizes reduce this to the
standard error of the job means. A single job has no block estimate (JSON null).

This is conditional on the available Cat library and the selected-track distribution.
It assumes independent jobs and does not resolve overlapping subsets, shared source
showers, selection/reference uncertainty or finite-library bias. A block error smaller
than the IID error is not proof that these effects are absent. Historical pool counts
cannot reconstruct which source tables were used.

## Remaining matched-response validation

1. Persist the actual selected Cat-table identities, metadata/hashes, code/conditions
   versions and seed before any new mixture. Do not infer table identity from counts.
2. Export a single mixed **full-step** sample plus an explicit all-event table containing
   `(run, subrun, event, pbtime, intensity)`. Preserve stepless events. Retain raw times;
   folding before export would bake the fast timing assumption into the stock control.
3. Feed exactly those steps and event conditions to stock and fast response, rather
   than comparing unrelated independent mixtures. The existing `CrvStepsFromTable`
   reader requires TTrees with exact branch types and particle rows; merely writing
   compact float32 Cat columns is not a valid adapter. Verify the built module and
   event-window/ProtonBunchTime producers before running it.
4. Check photon-arrival wrapping versus step wrapping with boundary inputs and optical
   delays, then compare charges, digis, pulses, coincidences and method F on the same
   complete event list. Use independent response seeds and retain paired-event errors.
5. Validate mixing efficiencies, intensity and species multiplicities against the
   production configuration separately. The output-frame audit does not certify the
   upstream Cat denominator inferred from generator rows.
6. Keep the BO deadtime objective on stock coincidences until the response/mixing
   closure gates are met. Do not reuse stale Cat extraction submission lists.

Tests: `python3 -m pytest -q crv/tests/test_veto.py crv/tests/test_veto_frameaudit.py`.

## Raw mixed-event replay pilot

`veto.replay` exports the raw mixture without response:

```bash
python3 -m crvfastsim.veto.replay --tables "$TABLES" --events 300 \
  --seed 20261025 --out "$NEW_REPLAY_ROOT"
python3 -m crvfastsim.veto.replay_fast "$NEW_REPLAY_ROOT" \
  --conditions "$CONDITIONS" --photon-dev "$PHOTON_DEV" --out "$NEW_FAST_FRAMES"
```

The mixer optional `return_events=True` returns its actual sampled pbtime/intensity
without drawing additional random numbers or changing the legacy four-value return.
The exporter writes `dump/events`, `dump/steps` and `dump/particles` as TTrees with
exact types expected by `CrvStepsFromTable`, including empty step/particle trees.
Particles are remapped densely per event; ancestry is synthetic (parent=-1, isprim=1),
so these products are response diagnostics, not a physics-truth sample.
NaN step times are preserved and counted: stock `CrvPhotonGenerator` explicitly skips
them, as does the fast photon-row finite-time filter. Infinite times and nonfinite
kinematics fail the export. Input tables, output ROOT file and exporter/mixer code are
hashed; versions and pool metadata are recorded in the JSON manifest.

`scripts/export_veto_replay_pilot.sh OUT INVENTORY CAT_PNFS [EVENTS] [SEED]` uses one
existing table per species from a completed basename inventory, stages only into a
per-job `/tmp` directory, exports, and removes its own staging directory. Original
pnfs locators are retained in `OUT.sources.txt`. This small subset is for wiring tests,
not a closure measurement or finite-library estimate; it never submits jobs.

The fast runner verifies the exported ROOT hash and retains the explicit all-event
list; unlike the generic file runner it does not infer frame membership from steps.
It rounds pbtime to float32, matching the stock event product. Both diagnostic arms
use exact MC bunch time for reconstructed time (no 0.5 ns production smearing).

### Stock arm: build required

`CrvReplayEventFromTable` is a new art producer reading `dump/events`. It publishes
`EventWindowMarker`, `ProtonBunchTimeMC`, and `ProtonBunchTime`, rejecting missing or
duplicate events and wrong branch types. Intensity is retained in the ROOT manifest
but not republished as an integer art product: mixing is already complete and CRV
response does not consume it. Existing SConscript libraries cover its dependencies;
this package has no CMakeLists file.

`CrvFastSim/fcl/replay_crv_run1a.fcl` uses the stock Run1A response/reco epilogs and
`CrvStepsFromTable`, with an EmptyEvent source for run 1430/subrun 0/events 1..N.
Override both `tableFile` entries with the same replay path and `source.maxEvents`
with its manifest count. The diagnostic fixes the window length at 1700 ns and
uses no reconstructed-bunch-time smearing; it is not a reproduction of production's
1675/1700 ns marker sequence. For the normal onspill CRV digitisation window the
response derives its bounds from bunch time, but this must still be runtime-checked.

The user must build the existing `crv/art` muse area before this producer is used.
The adapter has been built by the user and exercised in bounded stock replay jobs;
site-specific evidence and measured physics quantities are recorded in the internal wiki.
Start with one event, then ten, checking recorded event IDs/pbtime and output entries
before the full pilot. Use `veto.compareframes --dump` against the stock dumper output
only after both frame lists are verified complete. The full-Cat deficit remains open.

### Running and checking the stock replay

After the user confirms the build, set up that muse area and run serially:

```bash
bash "$CRVFS_CODE/scripts/run_veto_stock_replay.sh" "$AREA" "$REPLAY" "$NEW_OUTDIR" 1
# Repeat with new output directories for ten events and then the complete pilot.
python3 -m crvfastsim.veto.replay_check --input "$REPLAY" --stock "$NEW_OUTDIR/stock.root" \
  --events 1 --conditions "$CONDITIONS" --out "$NEW_OUTDIR/check.json"
# For the complete pilot, add --fast "$FAST_FRAMES" and set --events to the manifest count.
```

The driver requires the matching `MUSE_WORK_DIR`, new data-area output directory,
and a positive event count. It records the input table/source hash and backing area.
It never builds or alters a shared area. Run the checker with the ana Python plus
`PYTHONPATH=core:crv`, not an arbitrary Python bundled with art.

The checker opens the highest tree cycles, requires the exact requested event prefix
including empty events, float32 bunch times, onspill/1700 ns window, product-row counts
consistent with event counters, and equality of the dumped calibration/status/ordinal
arrays with the fast conditions file. Complete paired comparisons also require the
fast arm's input and conditions hashes and exact all-event ID list. Empty comparison
subsets serialize undefined fractions as JSON null using the common JSON writer.
The recorded calibration comparison does not independently certify the photon-yield
DB table; retain the photon-deviation file hash and stock conditions version too.

A successful replay contract is not a physics-closure test. A handful of coincidences,
a single Cat table per species, or a large KS p-value cannot certify a rate/shape model.

## Multiple recorded subsets and paired metrics

`veto.replay_subset` deterministically selects disjoint filename blocks from a given
inventory. `--exclude prior.json ...` excludes exact basenames in prior subset manifests;
use it when changing inventories. A completed inventory may be an old partial snapshot:
check the per-species available counts against the current census before claiming broad
coverage. Each selected file is read and hashed by the exporter; no directory listing on
pnfs is needed. Disjoint files do not prove independent underlying showers.

`export_veto_replay_subset.sh LIST OUT EVENTS SEED` stages only the named tables into
its own `/tmp`, exports the mixture, and removes its staging directory. The bounded
`run_veto_replay_block.sh AREA NEW_OUT BLOCK [EVENTS=3000]` runs one raw export, fast
response, and stock response serially in a user-built muse area. Required environment:
`CRVFS_CODE`, `CRVFS_DATA`, `ANA_PY`, `VETO_CAT_INVENTORY`, `VETO_CAT_PNFS`,
`VETO_CONDITIONS`, `VETO_PHOTON_DEV`. Optional `VETO_SUBSET_BLOCK` separates the subset
slice index from response/mix seeds, and comma-separated `VETO_SUBSET_EXCLUDE` supplies
prior manifests. It preserves muse's search path (needed for Database/connections.txt).

Run post-processing in a **separate ana-only shell**, not muse's Python/LD_LIBRARY_PATH:
loading pyxrootd under muse can select libstdc++ without the SciPy-required CXXABI.
`check_veto_replay_blocks.sh BASE CONDITIONS SELECTION REFERENCE block...` checks
retained outputs and writes a new paired report (`VETO_METRICS_OUT` overrides its path;
`VETO_REPLAY_EVENTS` defaults to 3000). It never reruns detector response. Paths named
`stock_retry` are supported for jobs whose initial stock setup failed before events.

`replay_fast --pulses PATH` captures fitted pulses from the **same** response draw via
an optional sink, without changing the response or random draws. The pulse output and
relevant runner code hashes are recorded. `veto.replay_metrics` consumes verified
checks and frame files, computes frame coverage with the frozen selection/card/grid,
and reports the standard error of the paired coverage difference and the job-block
error. Both are conditional on the selected-track distribution and sampled libraries;
neither is an unconditional finite-library uncertainty. Repeated input hashes or mixed
response variants cannot be pooled as independent frames.

The pulse diagnostic retains the strongest observed pulse per (event, channel), reports
T1 and T2 separately (Run1A channel boundaries), and compares matched PE/time quantiles
and PE sums. This is observed-channel matching, not truth-particle matching. Cluster
sector type alone cannot distinguish Run1A T1 from T2: both use sector type 1.

### Explicit stock shared-LUT yield diagnostic

`replay_fast --stock-shared-table-yields` is **diagnostic only**. Stock
`CrvPhotonGenerator` in Offline v13_33_00 and v13_39_00 reuses a `MakeCrvPhotons` object
when lookup-table filename strings match and skips the later sector's
`SetScintillationYield`. The object thus retains the first sector's yield. The fast
implementation normally honors each sector's configured yield. Run1A uses one LUT for
T1 and T2 but different configured yields, so these are distinct response contracts.
The flag copies the response config and substitutes the first yield per table; it does
not mutate `RUN_CONFIGS`, change `mixframes`, or change any BO evaluator/default.
Effective yields and the diagnostic flag are written to the sidecar.

Comparisons with this variant must reuse existing stock outputs and use a distinct
fast output tag. A change in photon multiplicity also changes subsequent RNG draws;
identical seeds do not make individual pulse fluctuations identical. Establish the
sector-level effect and overall deadtime closure on independent inputs before choosing
between stock compatibility and the intended per-sector physics. Fixing Offline or
changing production defaults requires separate review; this diagnostic does neither.
