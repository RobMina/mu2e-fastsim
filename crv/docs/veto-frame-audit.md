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
No C++ compilation, FHiCL runtime validation or stock replay has yet been performed.
Start with one event, then ten, checking recorded event IDs/pbtime and output entries
before the full pilot. Use `veto.compareframes --dump` against the stock dumper output
only after both frame lists are verified complete. The full-Cat deficit remains open.
