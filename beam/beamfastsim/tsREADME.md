# TS transport development (opt-in)

`tsreference`, `tsfield`, `tsgeometry`, `tsphysics`, `tstransport` and `tscompare`
are an independent charged-muon/pion development route. Nothing imports them from
the default beam or CE chains. Inputs must use the same geometry and field as the
stock sample. Run1B development and a Run1A development comparison exist; Run1A
held-out validation and a matched stock TS-only performance comparison remain open.

## Review-corrected development checkpoint (2026-11-06)

On the same 25,078 entrances / 100 development shards / 500,000 parent Events:
mu-minus **1,088/1,098 = 0.9909 +/- 0.0219**, mu-plus **8/15 = 0.5333 +/- 0.1633**.
Transport takes **187.215 s / 133.953 entrances/s**, one laptop CPU worker excluding
field loading. Settings/seed are unchanged, not fitted to yields. All inputs remain;
1,099 DS, 18,538 range-out, 210 mu-decay, 5,225 wall, 6 upstream, no path caps or
missing field. Positive deficit and 2,696 compact-ancestry gaps remain explicit.

Material bounds are rechecked through convergence and after boundary truncation;
unresolved proposals raise rather than sample over-cap columns. Entrance exits use
the bracketed plane intersection/aperture. Details: [transport guide](../docs/ts-transport.md).
Changed step sequences change RNG consumption; the saved outputs below are not
numerically identical to this run. The 12-run study predates the cap correction and
has not been rerun; do not use it as corrected-code convergence evidence. Local
TS suite: **409 passed**, both saved geometry fixtures supplied. Held-out untouched.

## Historical pre-review checkpoint (2026-11-06)

The completed Run1A development comparison uses 100 independent shards, 500,000
finished parent Events and 25,078 known-root mu/pi entrances. All zero-output
entrances and empty-source events are retained. Two nested entrances are represented
by their pion ancestors; 2,696 upstream compact-ancestry gaps remain, so this is not
complete primary-root closure. All 1,118 stock DS mu/pi records resolve to the cohort.

| Quantity | Fixed-seed, all-charge development result |
|---|---|
| DS mu-minus, fast / stock | 1,095 / 1,098; ratio 0.9973 +/- 0.0230 |
| DS mu-plus, fast / stock | 8 / 15; ratio 0.5333 +/- 0.1458; sparse deficit, not agreement |
| Transport cost | 188.811 s for 25,078 entrances; 132.820 entrances/s |
| Performance scope | One laptop CPU worker; field loading excluded; no matched stock TS-only or GPU benchmark |
| Regression evidence | Publication local TS suite: 344 passed, with both saved Run1A and Run1B GDML fixtures; not CI |

The measured transport improvement is 8.362x over the initial 1,578.929 s run and
2.880x over the first optimized 543.821 s run. All saved numerical arrays except
seconds agree exactly with the first optimized output. They are **not** all identical
to the initial output: preserving DS-plane snapping corrected five mu-minus wall
fates to DS; six terminal z coordinates changed by at most 2.27e-8 mm.

The explicit Run1A settings are RK4 `--step 2.5 --seed 20261105 --max-path 100000`,
0.1 rad angular cap, 0.25 mm material cap and 5% fractional mean-loss cap. The CLI
step default remains 5 mm. This historical checkpoint requires the pre-review code.
Rates use the appropriate source-charge cohort, not POT. Errors are paired-shard
estimates; diagonal shape chi2 includes yield and ignores bin covariance.

The bounded positive-source study below is complete and did not justify changing
these settings. Run1A held-out shards 100–199 remain unexamined. Positive-charge
losses, two study path-cap outcomes, compact upstream ancestry and matched stock
TS-only timing remain open. Run1B delivery assets must not be relabelled as Run1A.
The Run1A comparison is still a private development result, not a delivered
held-out validation. The route is opt-in; local tests do not imply CI success.

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
`tsrun1a_smoke.sbatch` takes site paths from environment variables; explicitly
approve simulation before submitting it. It does not build or install software.

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

## Run 1A reference campaign

`tsrun1a_campaign.sbatch` is an explicitly approved 1,000,000-POT campaign:
200 independent 5,000-POT shards, at most 32 concurrent, one CPU/12 GB/24 h limit
per shard. Indices 0–99 are development; 100–199 are held out before looking at
outcomes. Run1430, subrun920000+index, baseSeed920001+1000*index; seed spacing
exceeds the stock autoIncrement maxUniqueEngines100 range. The smoke is excluded.

Create a fresh `TS_CAMPAIGN_ROOT` with `config/`, `shards/`, `logs/`; freeze copies
of `tsrun1a_raw.fcl`, this batch script, and the tested `offline_uva.sh` in `config/`
before submission. Set `CVMFSEXEC_DIST` to the existing EL8 client and
`SBATCH_ACCOUNT` / `SBATCH_PARTITION` for the approved site allocation. Then:

```bash
sbatch --parsable -A "$SBATCH_ACCOUNT" -p "$SBATCH_PARTITION" \
  -o "$TS_CAMPAIGN_ROOT/logs/%A_%a.log" \
  "$TS_CAMPAIGN_ROOT/config/tsrun1a_campaign.sbatch"
```

Each task creates a new shard directory, appends only subrun/event-count/seed/art
filename overrides to its frozen FCL, and records the requested POT in shard.tsv.
The runner exit code propagates; automatic requeue and existing-output overwrite
are disabled. Both generation launchers force `OFFLINE_WARM_REPEAT=0`. Requested counts are not finished counts: inspect highest-cycle art
Events after successful completion, retain empty-source events, and report failed
or partial shards separately. Keep large raw outputs on UVA scratch, not in the
laptop mirror. Full ancestry remains in art; the compact diagnostic alone is not
a complete same-stage reference adapter. Runtime is measured, not extrapolated
from the smoke; the wall limit is a resource cap, not an ETA.

## Development-only Run 1A analysis

`python3 -m beamfastsim.tscampaign --root CAMPAIGN --generation-job JOB_ID
--index 0 --out census-000.json` checks a finished
5,000-POT shard with the compact census; indices outside 0..99 are rejected.
`python3 -m beamfastsim.tscampaign --combine census-*.json --out summary.json`
adds finished tree counts and keeps
unknown ancestry as null, not a false loss. Production files remain on UVA.
The direct uproot StatusG4 reader reports undecodable streamer values explicitly;
`tsstatus.fcl` is a stock analyzer-only re-read alternative, with no Geant4 producer.
Its batch wrapper requires `TS_GENERATION_JOB` for the same completion-log prefix;
no campaign job ID is built into either reader.

`python3 -m beamfastsim.tsrawreference --ntuple SHARD/tsrun1a_raw.root --gdml RUN1A.gdml --out reference.npz`
is the same-stage adapter, not the Run1B restart reader. It retains every forward
VD1 mu/pi track, first crossing state, all crossing records and all DS mu/pi rows.
Exact parent keys attach endpoints; flat tables preserve branching and unresolved
ancestry. An aligned one-endpoint reference is available only without unresolved
DS links, multiple endpoints or nested source tracks. Omitted ancestry upstream
of an already identified entrance remains explicitly reported. Full ancestry can
be recovered by stock all-species ReadVirtualDetector on the saved raw art, not by
setting navigateToParent (which only follows stage restarts).

`python3 -m beamfastsim.tsrawreference --combine reference-*.npz --out development.npz` merges the small
aligned references from development subruns920000..920099 and rejects duplicate
particle/event/shard keys. Check their census reports against finished art first.
For a one-lineage comparison use explicit `--root-lineages`: each known ancestor
entrance appears once, nested entrance states/root IDs remain in the output, and
DS descendants are attached to that ancestor. The saved per-track files are not
changed. All event IDs and zero-output roots remain. Multiple endpoints per root
or unresolved DS ancestry still raise; no endpoint is silently dropped. This is a
known-root projection, not proof of complete primary ancestry: compact omissions
upstream of an entrance can hide additional nesting and are counted separately.
Use the saved MDC2025ac/Offlinev10_47_00 Run1A GDML and fetch the matching maps:

```bash
python3 -m beamfastsim.tsfield field-run1a --release v10_47_00 \
  --field-config Offline/Mu2eG4/geom/bfgeom_no_ds_v01.txt
```

The fetch follows release includes before choosing maps. Run1B DSOff maps are not
a substitute. `python3 -m beamfastsim.tscompare --geometry-label Run1A` labels the new comparison without
changing the statistics or the default Run1B behavior. Do not tune on held-out data.

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

`python3 -m beamfastsim.tsstudy --reference development.npz --gdml matching.gdml --field-config config.txt
--offline-root offline --maps maps --variant baseline --seeds --out fresh-directory`
uses all 12,832 positive-charge inputs from the full 100-shard development reference,
including stock losses; full parent-event exposure remains saved. The fixed seed
list is 20261111..20261114; pass an explicit subset after `--seeds` if needed.
`half_material` jointly halves the material chord cap and fractional-loss cap;
`half_step` halves the field step alone. Neither changes the default settings.

Each setting/seed writes an NPZ and a JSON with yields, overlaps, paired-shard
errors and momentum/time histograms. `python3 -m beamfastsim.tsstudy --summarize run-files.json ...
--out summary.json` reports seed mean/SD/range and matched-seed setting differences.
Changing step size changes RNG consumption: matching seed labels are not common
random trajectories. Positive-only batching also changes the RNG stream relative
to the historical all-charge 8/15 result, so use the study's fresh baseline. Four
seeds diagnose variability; they do not establish significance or convergence.
Optional `--reference development.npz --gdml matching.gdml` with `--summarize`
adds per-ID fate/material tables for stock mu-plus survivors across saved NPZs.
Those survivor-selected rows are diagnostics, not a replacement rate denominator.

Historical pre-review yields for seeds 20261111, 20261112, 20261113 and 20261114:

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
outcomes. Saved summaries include capped IDs and their stock outcomes.

## Reproducing the local test coverage

From the umbrella checkout, use the project venv with the pinned core and this
`beam/` on `PYTHONPATH`. Both saved geometry fixtures must be supplied to reproduce
the recorded full TS coverage; without them the four geometry-dependent tests skip.

```bash
export PYTHONPATH=/path/to/pinned-core:$PWD/beam
export TS1_GDML=/path/to/ts_geometry_run1b.gdml
export TS1_RUN1A_GDML=/path/to/ts_geometry_run1a.gdml
/path/to/project/.venv/bin/python3 -m pytest -q beam/tests/test_ts*.py
```

The affected-suite selector also needs the umbrella `calo/` and the pinned tracker
on `PYTHONPATH`, plus `MU2E_OFFLINE_ROOT` for the optional stock DIO-table test.
Run it against the intended PR base with `python3 -m mu2efastsim.testselect
--also /path/to/pinned-core --run --isolate`. Test each final PR slice; local test
results and saved physics comparisons do not substitute for CI.

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
