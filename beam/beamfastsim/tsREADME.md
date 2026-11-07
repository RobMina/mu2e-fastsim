# TS transport development (opt-in)

`tsreference`, `tsfield`, `tsgeometry`, `tsphysics`, `tstransport` and `tscompare`
are an independent charged-muon/pion development route. Nothing imports them from
the default beam or CE chains. Inputs must use the same geometry and field as the
stock sample. Run1B and Run1A development comparisons and one frozen Run1A
held-out comparison are complete. Held-out is consumed, not a fresh tuning test;
a matched stock TS-only performance comparison remains unmeasured.

## Run1A development result (2026-11-06)

The corrected comparison uses 25,078 known-root entrances across 100 development
shards and 500,000 counted parent Events. All zero-output entrances and empty-source
events are retained. Two nested entrances are represented by their pion ancestors;
2,696 upstream compact-ancestry gaps remain, so this is not complete primary-root
closure. All 1,118 stock DS mu/pi records resolve to the cohort.

| Quantity | Corrected, fixed-seed development result |
|---|---|
| DS mu-minus, fast / stock | 1,088 / 1,098; ratio 0.9909 +/- 0.0219 |
| DS mu-plus, fast / stock | 8 / 15; ratio 0.5333 +/- 0.1633; sparse deficit |
| Transport cost | 187.215 s for 25,078 entrances; 133.953 entrances/s |
| Performance scope | One Apple M5 CPU core (single worker); field loading excluded; no matched stock TS-only, GPU or SU benchmark |

Fast fates are 1,099 DS, 18,538 range-out, 210 muon decay, 5,225 wall and 6 upstream;
there are no live, path-cap or missing-field outcomes. Settings and seed are
unchanged, not fitted to yields: RK4 `--step 2.5 --seed 20261105 --max-path 100000`,
0.1 rad angular cap, 0.25 mm material cap and 5% fractional mean-loss cap.
The CLI step default remains 5 mm.

Material bounds are rechecked through convergence and after boundary truncation;
unresolved proposals raise rather than sample over-cap columns. Entrance exits use
the bracketed plane intersection/aperture. See the [transport guide](../docs/ts-transport.md).
Changed step sequences change RNG consumption; equal seeds do not imply paired
trajectories. Rates use the source-charge cohort, not POT. Errors are paired-shard
estimates; diagonal shape chi2 includes yield and ignores bin covariance.

The historical pre-review output was 1,095/1,098 mu-minus (0.9973 +/- 0.0230),
8/15 mu-plus (0.5333 +/- 0.1458), and 188.811 s / 132.820 entrances/s.
Its optimization comparisons do not validate the corrected transport.

## Reference and transport (two-stage Run1B)

Use existing stock entrance files plus the downstream file that consumed them.
Keep every entrance, including those absent downstream. The stock
`ReadVirtualDetector` analyzer can read the recording `StepPointMC` collection
through `tvdStepPoints`: the resulting `nttvd` has the actual pre-step position,
momentum and time. These differ from SimParticle terminal states. `tsread.fcl`
shows the entrance configuration; for downstream change the three collection
labels to `TS5Filter`, `compressPVTS5` and `TS5Filter:Beam`. No Geant4 module runs. A fixed service seed permits analyzer initialization.

```bash
python3 -m beamfastsim.tsreference --source source.root --downstream downstream.root \
  --source-art source_inputs/*.art --out reference.npz
python3 -m beamfastsim.tsfield field --release v13_39_00
python3 -m beamfastsim.tstransport --reference reference.npz --gdml geometry.gdml \
  --field-config field/offline/Offline/Mu2eG4/geom/bfgeom_DSOff.txt \
  --offline-root field/offline --maps field/maps --shards SUBRUN_IDS \
  --step 5 --seed 20261105 --out fast.npz
python3 -m beamfastsim.tscompare --reference reference.npz --fast fast.npz \
  --shards SUBRUN_IDS --label development --out comparison
```

Set `PYTHONPATH` to this package's `beam/` and its pinned `core/`, or install the
packages in an isolated environment. The field fetch uses the released config's
map names, never nominal maps renamed as DSOff. Reference joins use event and
SimParticle keys and actual ancestry, not proximity or equal restart track IDs.
Parent `Events` tree counts retain empty source events per shard.

Choose step settings on development subruns, then report separate held-out
subruns once. Figures show absolute output per input cohort, paired per-shard
errors and pulls; the diagonal shape chi2 is descriptive with few shards, not a
calibrated goodness-of-fit claim. Sparse positive-charge endpoints do not support
a precise positive-charge efficiency measurement.

## Raw Run 1A smoke reference

`tsrun1a_raw.fcl` is a separate opt-in 100-POT test for released SimJob
MDC2025ac. It keeps stock Run 1A geometry, field and physics cuts, but writes
raw generator/counter/Geant4 products for every event without downstream output
selection. Raw art retains all species and full ancestry. The compact stock
`ReadVirtualDetector` diagnostic keeps mu+/mu-/pi+/pi-/proton particles and VD
hits; protons retain empty-source event bookkeeping. Actual DS-boundary Beam
StepPoints remain all-species. No custom recorder is used.
`tsrun1a_smoke.sbatch` takes site paths from environment variables and does not
build or install software.

After successful job completion, inspect the saved products with:

```bash
python3 -m beamfastsim.tssmoke --art sim.user.TSRun1ARaw.smoke.001430_00910000.art \
  --ntuple tsrun1a_raw.root --compact --expected-events 100 --out smoke.json
```

Add `--baseline-ntuple /path/to/full/tsrun1a_raw.root` for a same-seed rerun: every
retained scalar particle/VD record and all Beam records must match exactly.
Use the default full mode only for an all-species diagnostic.

This census checks event/proton counts and retained identities, and reports VD1
forward recrossings. Missing selected-species parents are errors; omitted-species
parents are explicit genealogy gaps. An unresolved DS endpoint makes the definite
source-loss count null, not a false loss. Full branching ancestry requires the raw
art reference; the compact ntuple alone is not sufficient. With no DS mu/pi, source
losses are unambiguous. This is not a physics comparison.
A same-stage raw reference must not be passed to the two-stage restart reader
without an explicit adapter; source recrossings and branching need interpretation.

## Run1A reference campaign and cohorts

The Run1A raw campaign has 200 independent 5,000-POT shards, with indices 0–99
for development and 100–199 held out. Requested POT is not finished exposure;
census counts come from completed art Events trees, including empty-source events.
Both generation launchers disable warm repeats. See the [Run1A reference guide](../docs/ts-run1a.md)
for campaign, census, explicit cohort selection and the stock status reader.

`tsrawreference` is a same-stage adapter, not the Run1B restart reader. It keeps
first forward VD1 mu/pi states, all crossing records and actual DS2Vacuum Beam
StepPoints. Exact event/parent-key joins preserve branching and zero-output inputs.
Explicit `--root-lineages` projects nested entrances onto known ancestor entrances;
nested states remain saved. Unresolved DS ancestry or multiple endpoints raise,
rather than dropping rows. Compact upstream omissions can hide additional nesting:
known-root projection is not proof of full primary-root closure. Full ancestry
requires the raw art products.

Use the saved MDC2025ac / Offline v10_47_00 Run1A GDML and matching field maps:

```bash
python3 -m beamfastsim.tsfield field-run1a --release v10_47_00 \
  --field-config Offline/Mu2eG4/geom/bfgeom_no_ds_v01.txt
```

The fetch follows release includes before choosing maps. Run1B DSOff maps are not
a substitute. `tscompare --geometry-label Run1A` changes the comparison label, not
the statistics or default Run1B behavior. Cohort merges default to development;
held-out indices require explicit `--cohort heldout` and cannot be mixed with it.
The completed held-out sample is consumed, not a fresh tuning test.

## Saved-output diagnosis and performance

`python3 -m beamfastsim.tsdiagnose --reference development.npz --fast fast.npz --out fresh-directory`
joins exact input IDs and writes per-charge/species fate contingencies, stock-only
and mu-plus-union terminal-state tables, and associations with saved ancestry gaps.
The CLI accepts Run1A development subruns only. Optional `--gdml matching.gdml`
classifies terminal positions in selected materials, not full interaction histories.
Fast range-out for a stock survivor locates a mismatch; it does not establish
excess energy loss as its cause.

Profile the existing transport command with `python3 -m cProfile -o run.prof -m
beamfastsim.tstransport ...`; inspect with `python3 -m beamfastsim.tsbench --profile run.prof`. Compare
same-input, same-seed outputs with `python3 -m beamfastsim.tsbench --reference baseline.npz --candidate
candidate.npz --out comparison.json`. Every saved array except elapsed seconds
is checked; exact means numerical equality including matching NaNs, not NaN payloads.
Small profiled batches are hotspot diagnostics, not full-cohort runtime predictions.

Field queries skip maps after coverage and omit already-covered rows; interpolation
and map priority are unchanged. Each repeated advance reuses its entry field, and
DS/wall bisections advance only crossing rows. Field evaluators must be stateless.
This also preserves a snapped DS endpoint when a different row crosses a wall;
whole-batch re-evaluation could round the DS row outside a closed envelope.
The exact optimization comparison refers to the preserved pre-review outputs only.

The unit-grid interpolation gathers eight corners together and accumulates in the
same order as core; independent-core tests require bitwise field agreement. A
conservative envelope/map clearance can skip quarter-step probes only when every
fractional RK stage and endpoint is inside both domains. For h, p and global Bmax,
A=h*K*Bmax/p gives radius h*max(1+A/2+A^2/4,1+A/2+A^2/6+A^3/24), plus rounding padding.
This follows the RK direction-stage norm bounds, not the nominal arc length alone.
The field bound covers every map including priority transitions; reflected maps
are continuous in coverage across y=0. Material columns and full-step/end checks
remain active. Unsupported safety objects use the ordinary probe path; Python
`boundary_safety=False` permits an exact same-seed regression. Material shortening
and boundary column recalculations target changed chords; a final cap failure retries
the boundary proposal and flags before any material sampling.

## Bounded positive-source development study

This development study uses all 12,832 positive-charge inputs, including stock
losses, from the 100-shard reference; all 500,000 parent-event IDs remain saved.
`half_material` jointly halves the material chord and fractional-loss caps;
`half_step` halves the field step alone. Neither changes the default settings.
Changing step size changes RNG consumption: matching seed labels are not common
random trajectories. Positive-only batching also changes the RNG stream relative
to the all-charge 8/15 result, so comparisons use the study's own baseline.
Four seeds diagnose variability; they do not establish significance or convergence.

### Corrected transport (2026-11-06)

All 12 runs use the same inputs, four seeds and variants, with the enforced
material caps and boundary corrections. Counts for seeds 20261111 through 20261114:

| Variant | DS mu-plus counts | Mean / sample SD |
|---|---|---|
| `baseline` | 12, 11, 15, 10 | 12.00 / 2.1602 |
| `half_material` | 10, 9, 10, 10 | 9.75 / 0.5000 |
| `half_step` | 14, 14, 12, 11 | 12.75 / 1.5000 |

Every run retains 12,832 positive inputs and 500,000 parent-event IDs, with zero
live, path-limit or missing-field outcomes. Stock has 15 mu-plus endpoints; no
stock or fast pi-plus endpoint.
Halving material controls does not remove the deficit. Halving the field step
changes counts by +2, +3, -3, +1 relative to the baseline seed labels, not a
convergent sequence. The shared-stock-survivor mean is 10.25 baseline / 10.00
half-step; fast-only mean is 1.75 / 2.75. The small total-yield gain is not evidence
of recovering missed stock survivors. Do not substitute the 15/15 seed for the
fixed all-charge result or choose settings by proximity to stock.

Survivor-selected terminal material labels are diagnostics, not stock material
histories or proof of excessive energy loss. Keep the documented
2.5 mm / 0.25 mm / 5% development point; numerical/model convergence remains open.

### All-charge joint-tightening sensitivity (2026-11-06)

One full-development candidate halves field/material/loss caps to
1.25 mm / 0.125 mm / 2.5%, retaining all 25,078 inputs in original order, 500,000
parent event IDs and seed 20261105. The corrected baseline is reused, not rerun.
DS mu-minus changes 1,088 -> 1,073 (stock 1,098); mu-plus 8 -> 9 (stock 15).
Tighter/baseline ratios are 0.98621 +/- 0.01985 and 1.125 +/- 0.437 respectively.
Mu-minus source-fraction change is -0.12249 +/- 0.17769 percentage points.
Paired-shard setting-difference momentum/time diagonal chi2 are 4.643/11 and
22.352/21 for mu-minus; 3.347/3 and 6.351/8 for mu-plus. Maximum absolute bin pull
is 1.56; no significance from scanning bins. Both fast settings have zero muon
histogram flows, live/path-limit/missing-field outcomes and DS pi-plus; 3 DS pi-minus.

No clear population instability is resolved at this sample size. This supports
keeping the baseline for the subsequently completed frozen held-out test, not
convergence or precision agreement of sparse positives. Equal seeds are not common-random
histories: 267 mu-minus baseline-only / 252 tighter-only endpoints. Joint controls
cannot isolate a numerical cause; no setting is selected by closeness to stock.
Tighter transport cost is 398.133 s (62.989 entrances/s), one Apple M5 CPU core (single worker),
excluding field load; concurrent tests ran during part of this measurement. Keep the original
baseline benchmark and development headline. Held-out was unopened at this checkpoint.

`tssensitivity` compares saved baseline and candidate outputs against a shared
reference; it does not run transport or choose settings. The CLI requires
`--reference`, `--baseline`, `--candidate` and `--out`. See the
[saved-output comparison guide](../docs/ts-sensitivity.md) for usage and statistics.

### Optional history diagnostics

Reusable selected-lane trace collection, explicit-target stock crossing extraction
and material/x-plane analysis helpers are documented in the
[history guide](../docs/ts-history.md). Sparse stock crossings are not material
histories; fast x-plane samples are not validated VD hits. These diagnostics do
not establish numerical convergence or justify retuning.

### Historical pre-review study

Historical yields for seeds 20261111, 20261112, 20261113 and 20261114:

| Variant | Field / material / fractional-loss caps | DS mu-plus counts | Mean / sample SD |
|---|---|---|---|
| `baseline` | 2.5 mm / 0.25 mm / 5% | 9, 10, 11, 11 | 10.25 / 0.9574 |
| `half_material` | 2.5 mm / 0.125 mm / 2.5% | 11, 11, 9, 9 | 10.00 / 1.1547 |
| `half_step` | 1.25 mm / 0.25 mm / 5% | 13, 11, 12, 10 | 11.50 / 1.2910 |

Stock has 15 mu-plus and zero pi-plus endpoints; no fast pi-plus survives in these
runs. The four-seed SD is not a rate measurement error. The joint material-cap
change has no monotonic improvement; the field-step result is inconclusive for
convergence. Settings were not chosen by closeness to the stock yield.

Two inputs reach the 100 m path cap: `baseline` seed 20261113 and `half_step` seed
20261112 each have one. Both are stock non-survivors. They remain in the denominator
with explicit `path_limit` fates, not physical-absorption labels; a later fast DS
endpoint remains possible for each. All 12 runs have zero live or missing-field
outcomes.

## Frozen Run1A held-out comparison (2026-11-06)

The once-only held-out comparison (yields, spectra, fates and cost) is recorded in one place: the
[transport guide](../docs/ts-transport.md#frozen-held-out-result-2026-11-06). The held-out set is
consumed, not a fresh tuning test.

## Tests

Run the TS tests under `beam/tests/` with the pinned core and `beam/` on
`PYTHONPATH`. Set `TS1_GDML` and `TS1_RUN1A_GDML` to matching saved geometries
for the four geometry-dependent tests; synthetic tests need no collaboration data.
Local results and saved physics comparisons do not substitute for CI.

## Model and limitations

- Batched core RK4/helix field stepping; maps use trilinear interpolation and the
  declared Y reflection. Every accepted integration stage checks map coverage;
  angular steps are capped at 0.1 rad. Fixed first-containing map precedence
  differs from the stock shared last-map cache in overlapping domains.
- TS1–TS5 cylinders/tori and their material daughters from the supplied GDML.
  Analytic chord intersections preserve collimator holes and thin absorbers.
  Walls terminate transport; cryostat return and other upstream regions are not
  transported. Quarter-step envelope samples detect and localize wall exits;
  excursions between samples remain a step-size approximation. The downstream
  boundary is not relabelled as a VD.
- Core ionization/Urban fluctuations and cumulative Highland scattering,
  integrated with small material and fractional-loss steps. Chords approximate
  curved paths; interactions within one step are not ordered by surface. Time
  integration uses step-entry momentum. Range-out below 0.2 MeV is approximate.
- Pion two-body decay with exact masses and isotropic rest-frame direction;
  sampled proper lifetime is integrated along the flight. No polarization,
  pion nuclear interactions, stopped-pion daughters, muon capture or muon-decay
  electrons. Mass-substituted muon ionization is approximate for spin-zero pions.

Physics disagreement is reported, not corrected by fitting scales to the stock
result. Stock full-POT timing cannot serve as a same-boundary TS-only benchmark.
