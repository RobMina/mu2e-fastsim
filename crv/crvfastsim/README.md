# crvfastsim — Geant4-free CRV cosmic simulation and its validation

One Python package with a single CLI, for the KPP (extracted) CRV cosmic sample. It follows the conventions of
`crvextrap` (`RobMina/mu2e_track_crv_mcresponse`): definitions live once in shared modules, every question is a
subcommand, hot paths are array code, and batchable analyses use `--emit` / `--merge`. The plan, goals and thresholds
are in [`../docs/plan.md`](../docs/plan.md); this file is the living reference for the code.

```
.venv/bin/python3 -m crvfastsim <subcommand> <inputs> [--compare REF ...] [--outdir DIR] [--emit X.json | --merge]
```

`crvfastsim` builds on `mu2efastsim` (the `core/` submodule of the umbrella `mu2e-fastsim`: energy loss, materials,
world, generic crossings/transport/tables, statistics, plots, key matching, the IRI job tools). Put both on the path:
from the umbrella root, `export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker` (what `site.env` does), or
`pip install -e core -e crv` in a venv. Laptop: `~/mu2e/.venv` (numpy, pandas, uproot, matplotlib, jax CPU, pytest).
GPVM: `source setupmu2e-art.sh && pyenv ana` (no JAX: the validation subcommands do not need it). Polaris: the project
venv on Eagle (`core/iri/scripts/setup_venv.sh`, `jax[cuda12]`).

---

## 1. Samples

| name | what | where |
|---|---|---|
| `g4ref_v1` | G4 reference: stock `ExtractedCRY` (run 1400, Musing `SimJob MDC2025ay`), 20 x 50,000 generated, 50,008 accepted, 15.587 s live time | art: `$CRVFS_PNFS/g4ref_v1/{dts,gen}.*.art`; tables: `$CRVFS_DATA/g4ref_v1_tables/{crvsteps,gen}.*.root` |
| profile | the 40,000-event GPVM profiling run (2026-10-15) | `$CRVFS_DATA/profile/` (`timing_*.db`, `crvsteps.profile.root`) |

Producer of the reference tables: `art/dump_g4ref.sh` (the `CrvStepTableDumper` art module, `art/README.md`).

## 2. The CrvStep table  → `tables.py`

TFileService directory `dump/`. Keys `(run, subrun, event)` are shared by both arms: the fast sim reads the `gen`
table and keeps the triple, so comparisons are event by event.

| tree | one row per | columns |
|---|---|---|
| `steps` | CrvStep | `nstep` (steps in the event), `bar`, `sp` (SimParticle key), `edep` (visible, MeV), `t0`, `t1` (ns), `x0,y0,z0,x1,y1,z1` (mm, Mu2e), `px0,py0,pz0` (MeV/c), `p1` (\|p\| at end), `len` (true path, mm) |
| `particles` | SimParticle of an accepted event | `key`, `parent` (-1 = none), `pdg`, `ccode`/`scode` (ProcessCode ids), `isprim`, `genidx`, `sx..ez`, `spx..ee`, `st`, `et` |
| `gen` | CRY primary, every generated event (gen stream) | `idx`, `pdg`, `genid`, `x,y,z,px,py,pz,e,t` (time after `cosmicTimeOffset`) |
| `subruns` | subrun | `livetime` (s, CRY `timeSimulated`), `primaries` |

Why a dumper and not uproot on the art file: uproot rejects `CrvStep` (strided-size check; 174 bytes per record on
disk) and cannot read the memberwise-streamed `SimParticle`/`GenParticle`.

## 3. Truth identity  → `tables.primary_muon_crossings`, `analyses/census.py`

Matching is **by truth-particle identity, never by time or distance** (a Mu2e analysis rule).

- **Primary muon** = the SimParticle with `isprim == 1` and \|pdg\| == 13.
- **A crossing** = the sum of that particle's CrvSteps in one bar: G4 starts a new CrvStep after each boundary exit,
  so `edep` and `len` are summed and the earliest `t0` kept.
- **Origin classes** (census): every step's particle is walked to its primary through `parent` (vectorised: one
  parent lookup per pass over all particles, bounded by the deepest chain), giving `mu_primary`, `mu_descendant`,
  `nonmu_primary` and `nonmu_descendant`.

## 4. Geometry  → `geometry.py`, `crossings.py`

- `from_offline_config(counters)` re-implements `GeometryService/src/CosmicRayShieldMaker.cc` on the Offline counters
  file. Bar index = creation order (sector → module → layer → counter). A module move is a parameter edit.
  `find_counters` resolves `Offline/...` paths through `MU2E_SEARCH_PATH`, the pinned CVMFS release, then the
  laptop mirror.
- `from_gdml` / `gdml_global_placements` read the Geant4 GDML export (translations only; V0 found no rotations on
  any path to a bar).
- `crossings.slab_crossings` is the JAX slab method over N tracks × 576 axis-aligned bars (float64 enabled).

## 5. Validation thresholds (agreed 2026-10-15)

| level | subcommand | thresholds |
|---|---|---|
| V0 geometry | `v0check` | every bar centre and half length within 1 um; same ids at every index; no rotations |
| V1 muon crossings | `v1compare` | per group (EX, T1, T2, M): crossings/event 1 %; path mean 0.5 %; visible-energy mean and MPV 1 % (MPV allows 2 sigma of its fit error); path and visible-energy binned ratios chi2/ndf ≤ 1.5 over 20 bins (user, 2026-10-16: the earlier per-bin ±2 %/±3 % after 2 sigma failed identical samples 22 % of the time; the band values are still reported); t0 bias ≤ 0.1 ns |
| V2 response (path A) | `v2compare` | per group: seq-0 pulse PE median 2 % (2 sigma allowed); PE spectrum 5-150 PE in 20 equal-content bins, ±5 % after 2 sigma; coincidence clusters per event 1 % (2 sigma allowed); per-bar rate = events with a ≥ 5 PE pulse in the bar, all 576 bars, chi2/ndf ≤ 1.5 and no bar beyond 5 sigma (user, 2026-10-16: counted once per event and bar, since counting each SiPM's pulse gave ~4 correlated entries per crossing). Per-layer efficiency of truth crossings is a diagnostic only (the agreed observable uses extrapolated tracks and the gap width D) |
| V3 tier 1b (step level) | `tier1b-check` | per group, held-out shards: descendant visible energy per muon crossing (same bar, other bars), non-muon visible energy and bar hits, within 10 % relative (plan V3); descendant positions in the bar (no midpoint on a face) |
| V4 | not written yet | `docs/plan.md` section 5 |

**Tier 1b library** (`tier1b.py`). Tier 1a writes only the primary muon. Tier 1b adds what G4 also makes: the muon's
descendants (deltas above the production cut, decay electrons) and the non-muon CRY primaries with their showers,
resampled from G4 CrvStep tables of **training** shards and checked on others.
- *Descendants*: each fast crossing draws a G4 crossing from the same (sector group, momentum, path-length) bin; edges
  are per-group quantiles, path length finer in the tail (1/3, 2/3, 90, 97 %). Same-bar descendants are offsets from
  the muon entry, rotated to the fast direction; if they fall outside the fast bar they keep the rotated position
  along the bar and take their G4 position across it (thickness, width). Other-bar descendants keep the bar structure
  (G4 bar offset and in-bar position, shifted along the bar with the muon entry). `fit_inside` keeps every midpoint
  0.1 mm inside the box.
- *Non-muon primaries*: each non-muon CRY primary of the event reaches the CRV with the library probability of its
  (pdg, kinetic-energy) bin and then gets a recorded G4 shower of that bin (absolute positions; bar indices of the
  library event, so a geometry move does not move them).
- *Build*: `tier1b-build crvsteps.*.root --gen gen.*.root --out lib.root`, one file at a time (a 5-file production
  build peaks at 5.2 GB). Production dts (`art/dump_prod_as.sh`) have no gen stream: pass `--generated N`
  (500 k per file, `dh.gencount`) and gen tables of the same CRY configuration for the reach denominator.
- *V2 arms*: tier-1b showers sit in events G4's filter rejected, so the candidate arm runs on the **gen stream**
  (`run_v2_arms.sh <k> fast1b <dir> <tag> gen`) and `v2compare --generated N` normalises per generated event.
- *Diagnostics*: `v2origin` splits reco pulses (single-pulse channels) and light yield (PE per MeV of visible energy)
  by deposit origin (muon / descendant / non-muon), which located the face-placement bug.

**V2 arms.** `scripts/run_v2_arms.sh <shard> fast|g4mu` runs the stock extracted digitisation and CRV reco
(`CrvFastSim/fcl/dump_crvreco_from_dts.fcl`) on the G4 dts file of the shard, once on G4's own CrvSteps (`ref`) and once
on a CrvStep table injected by `CrvStepsFromTable` (`fast`: the transport output; `g4mu`: `g4muon` = G4's steps cut to
the primary muon). `fast` vs `g4mu` is the like-for-like test of tier 1a through the response; `g4mu` vs `ref` measures
what tier 1b (deltas above the cut, non-muon component) must add; `fast` vs `ref` is the full V2. The art job sees only
the events G4 accepted, so fast crossings in G4-rejected events do not enter V2. Tables read by the art module must be
TTrees (`tables.write(as_ttree=True)`); RNTuple tables fail at construction with "no dump/particles or dump/steps".

**V1 event universe: the union of both arms.** Each arm holds every event in which the primary muon crossed a bar:
the G4 `PrimaryFilter` CRV term accepts any CrvStep above 0.1 MeV (so a G4-rejected event has no crossing), and the
fast arm writes every primary muon. An event absent from one arm therefore has zero crossings there. Restricting to
the events common to both arms (the first version) dropped the fast crossings in events G4 rejected (521 of 37,875
fast events on `g4ref_v1`, 2026-10-16: muons G4 stopped or deflected in material the fast world lacks), hiding a
real crossing excess. The MPV is a Moyal fit to the 15-70 % quantile core with its fit error; the check allows
2 sigma (a peak-finder moved by ±4 % with its smoothing window on the 16 k tagger crossings).

## 6. CLI reference

Each subcommand: `… -m crvfastsim <cmd> <inputs> [--outdir DIR]`. Default `--outdir` is `plots/<cmd>` (git-ignored).
Outputs are a stats JSON (`plots.write_json`: NaN → null) plus figures.

| Subcommand | Computes | Output |
|---|---|---|
| `g4cost` | cost of the stock G4 chain from TimeTracker SQLite: accepted fraction, `g4run` on accepted vs rejected events, CRV response, full digi, per module (vectorised pandas pivot; regression-identical to the 2026-10-15 loop script) | `g4cost_stats.json` |
| `census` | what reaches the CRV by origin class: visible-energy and step shares, per sector group, non-muon share by primary pdg, steps and bars per event, muon-only event fraction. Batchable | `census_stats.json` |
| `transport` | fast muon transport on the CRY primaries of `gen.*.root` tables, writing `fast<stage>.*.root` tables with the same keys. `--stage A`: straight lines in air, Geant4-matched restricted loss + Urban fluctuations + Birks in the bars (`eloss.py`). ~2 s per 50 k-event shard on a GPVM (numpy backend) | `fast<stage>.*.root` |
| `v0check` | V0: bar geometry vs the GDML and the constructCRV printout. `--g4log`, `--table CSV` | `v0_check.json` |
| `v1compare` | V1: primary-muon crossings fast vs G4 (`--compare` = G4 tables), fixed-edge histograms, events restricted to those in both arms. Batchable | `v1compare_stats.json`, `v1compare_{len,edep}.png` |
| `g4muon` | G4 CrvStep tables cut to the primary muon's own steps, typed like the fast tables (the `g4mu` V2 arm) | `g4mu.*.root` |
| `v2compare` | V2: two CrvRecoTableDumper arms (`--compare` = reference); `--steps`/`--compare-steps` add the layer-efficiency diagnostic; `--generated N` for rates per generated event. Batchable | `v2compare_stats.json`, `v2compare_pe.png` |
| `secondaries` | tier-1b design input: G4 steps beyond the primary muon (descendants by process, birth place and bar; non-muon primaries by pdg and energy) | `secondaries_stats.json`, `secondaries_extra.png` |
| `tier1b-build` | tier-1b library from G4 CrvStep tables (`--gen`, `--out`, `--generated`) | `lib.root` + `lib.json` |
| `tier1b-apply` | add library descendants and showers to fast tables (`--library`, `--seed`) | `fast1b.*.root` |
| `tier1b-check` | V3 step-level closure per group vs G4 tables of the same shards (`--compare`) | `tier1b_check.json` |
| `v2origin` | V2 diagnostic: pulses per PE bin and light yield by deposit origin, two arms (`--steps`, `--compare`, `--compare-steps`, `--generated`) | `v2origin_stats.json` |

**Batch + merge** (same contract as crvextrap §7): workers `--emit batch_NN.json` on a slice of files (one job per
GPVM), then `--merge "batches/*.json"` locally. Merged counts are sums, so the merge equals a single pass.

## 7. Vectorisation contract

No Python loops over events, steps or particles; loops over files, sector groups and fixed small sets only. Shared
reductions are pandas `groupby` / `merge` and `np.histogram` / `np.bincount` on whole arrays. A rewrite is
regression-checked against the previous output before it replaces it (done for `g4cost` and `census` on 2026-10-16).

Measured (a Mu2e interactive machine, `pyenv ana`): `census` on the 20 `g4ref_v1` tables (44,737 events, ~1.6 M steps) 3.9 s;
`v1compare` self-test on one table 2.7 s; `g4cost` on the 40 k-event profile 6.6 s.

## 8. Package layout

```
crvfastsim/
  __init__.py   package doc + version
  cli.py        argparse subcommand dispatch (+ __main__.py)
  geometry.py   CRV bar geometry (Offline counters file, GDML readers, find_counters)
  crossings.py  CRV bar boxes on mu2efastsim.crossings (JAX if installed, else numpy; identical in float64)
  transport.py  tier-1a muon transport to the CRV bars (stages A-C); shared MSC and slab loss from mu2efastsim.transport
  tables.py     CrvStep tree on mu2efastsim.tables; primary-muon crossings
  tier1b.py     tier-1b library: build (G4 tables) and sample (fast tables)
  analyses/     g4cost, census, transport, v0check, v1compare, g4muon, v2compare, v2origin, secondaries, tier1b
  reco/         fast CRV reconstruction (section 10; subcommands reco-*)
mu2efastsim (core/): eloss, materials, world, crossings, transport, tables, stats, plots, match
scripts/        run_transport_v1.sh (transport the 20 g4ref_v1 shards + V1, one GPVM, ~40 s);
                run_v2_arms.sh (V2 path A arms for one shard, ~1 min per art job)
tests/          pytest unit tests (synthetic inputs; geometry tests need only the counters file)
```

## 9. Results so far

| check | sample | result |
|---|---|---|
| V0 | v04 GDML (2026-10-16) | **pass**: 576 bars, max centre residual 3.6e-12 mm |
| census | `g4ref_v1` | run `crvfastsim census`; shares not quoted here |
| V1 | self-test (G4 vs itself) | pass by construction; harness check only |
| V2 | path A, all 20 `g4ref_v1` shards (50,008 events), `fast` (stage C) vs `g4mu` (2026-10-16) | **pass** in every group: PE median within 0.5 % (EX -0.09 %, T1 -0.06 %, T2 -0.12 %, M -0.48 %), PE spectrum within 2.5 %, clusters per event within 1.8 % (all within 1 sigma), per-bar chi2/ndf 0.33 (max pull 2.6). Layer-efficiency diagnostic within 0.7 points |
| V2 | path A, `fast1b` (stage C + tier 1b, library from 20 production files = 10 M generated events) vs `ref`, held-out `g4ref_v1` shards 11-20 (500 k generated), candidate on the gen stream (2026-10-19) | **EX, T2, M pass; T1 fails the PE spectrum** (worst bin 6.8 % after 2 sigma, limit 5 %). PE median EX -1.4 %, T1 -1.8 %, T2 -1.1 %, M +0.05 %; clusters per generated event within 0.8 % (M -3.2 ± 3.3 %); per-bar chi2/ndf 0.41. Light yield per MeV: muon and non-muon within 0.5 %, descendants 2-3 % low (M 6 %). The T1 residual is in muon-dominated pulses (fewer at 5-20 PE, more at 20-50 PE): tier 1a, not the library |
| V3 | `tier1b-check`, same library and shards (2026-10-19) | descendant energy in the muon's bar within 3 % of G4 (q99 of the bar total within 0.3 MeV); non-muon energy 0.94-0.99 of G4 per group; descendants in other bars 40-70 % of G4 in EX/T1/T2 (about 0.03 MeV per crossing short), 20 % in M |
| V2 tier-1b history | same shards (2026-10-16..19) | worst spectrum bin EX/T1/T2 after 2 sigma: no tier 1b 80/75/84 %; first library 11/9/11 %; other-bar placement 9/6/8 %; tail bins 7/8/5 %; no step on a face 5/6/3 %; G4 transverse position 4/8/4 %; production library 4/7/4 % |
| V2 | path A, `fast` vs `ref` (full G4) (2026-10-16) | **fails**, as expected without tier 1b: `g4mu` vs `ref` shows the same deficit, so it comes from the missing secondaries, not the transport or the table injection. Tier 1b is required for V2 |
| V1 | transport stage C (B + GDML world: concrete, overburden, steel, Al; sliced range-out; height-band MSC) vs `g4ref_v1` (2026-10-16) | **pass** in every group. Crossings EX -0.26 %, T1 +0.87 %, T2 -0.26 %, M -0.85 %; every path and visible-energy mean within 0.43 %, MPV within 0.7 %; binned-ratio chi2/ndf 0.31-1.08. 10 s per 50 k-event shard (GPVM, numpy) |
| P1 | transport CPU cost, one GPVM core (numpy backend, mu2egpvm05 shared with other users' jobs; CPU time, best of 2), shard 2 = 50,000 generated, 1,976 with CRV crossings (2026-10-16) | stage A 0.74, B 1.52, **C 2.34 ms per event with crossings** (29 / 60 / 92 us per generated event); world load 3.3 s once per job; RSS 1.8 GB. Against the stock G4 stage 433 ms per accepted event: C is ~185x. P1 target (<= 2 ms per accepted event) is missed by 17 % for C on CPU; the cost is per generated muon (MSC, world tracing, crossings), so the GPU port is the planned lever |
| V1 | transport stage B (air + Highland MSC to a plane above the CRV) vs `g4ref_v1` (2026-10-16) | **pass** in every group (crossings within 0.92 %, means within 0.70 %) |
| V1 | transport stage A (straight lines in air, no MSC) vs `g4ref_v1`, all 20 shards, 43,422 common events (2026-10-16) | EX, T1, T2 **pass** every check (EX: crossings -0.84 %, path +0.01 %, edep -0.22 %, MPV -0.03 %); taggers **fail** (crossings -1.0 %, path -0.54 %, MPV -1.6 %): ~24 % of tagger crossings land on a neighbouring bar because the straight line misses G4's track by 5 mm (median) to 40 mm (p90) after ~13 m without multiple scattering, against a 105.5 mm wide tagger crossed edge-on. Matched crossings agree (edep +0.01 %). Stage B adds MSC and the world material |

---

## 10. Fast CRV reco → `reco/` (prototype, 2026-10-16)

Goals R1-R6 and thresholds V-R1..V-R5 (agreed 2026-10-16) are in `docs/fastreco.md`.
Two paths from SiPM charges to reco pulses are prototyped and compared on fidelity and cost:

| module | content |
|---|---|
| `reco/reftables.py` | readers for the `CrvRecoTableDumper` trees (`crvdump/`: events, charges, digimc, digis, pulses, clusters, calib) |
| `reco/pulsefit.py` | batched port of `MakeCrvRecoPulses` + `CrvRecoPulsesFinder`: peak finder, fit window, Gumbel fit by fixed-iteration LM in Minuit's sine-mapped coordinates, failed-fit rules, subtract-and-repeat, digi grouping |
| `reco/stockfit.py` | Offline's own `MakeCrvRecoPulses` through PyROOT (the oracle; needs a built muse area) |
| `reco/waveform.py` | batched port of `CrvWaveformsGenerator` (ZS) + `MakeCrvWaveforms` + `MakeCrvDigis` |
| `reco/direct.py` | direct charge -> pulse model (v0: conditional empirical resampling per charge/spread bin) |
| `reco/compare.py` | V-R1/V-R2 matching and summaries, KS |

Subcommands of the package CLI: `python3 -m crvfastsim reco-vr2|reco-vr2stock|reco-vr2jax|reco-bench|reco-vr1|
reco-direct|reco-cost|reco-inspect REF.root ...` (alias `python3 -m crvfastsim.reco <cmd>`); key joins and the KS test
come from `mu2efastsim.match`. Tests: `tests/test_reco_pulsefit.py`.
