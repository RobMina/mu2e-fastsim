# crvfastsim — Geant4-free CRV cosmic simulation and its validation

One Python package with a single CLI, for the KPP (extracted) CRV cosmic sample. It follows the conventions of
`crvextrap` (`RobMina/mu2e_track_crv_mcresponse`): definitions live once in shared modules, every question is a
subcommand, hot paths are array code, and batchable analyses use `--emit` / `--merge`. The plan, goals and thresholds
are in [`../docs/plan.md`](../docs/plan.md); this file is the living reference for the code.

```
.venv/bin/python3 -m crvfastsim <subcommand> <inputs> [--compare REF ...] [--outdir DIR] [--emit X.json | --merge]
```

Run from the repository root (`mu2e-crv-fastsim/`) so the package imports. Laptop: `~/mu2e/.venv` (numpy, pandas,
uproot, matplotlib, jax CPU, pytest). GPVM: `source setupmu2e-art.sh && pyenv ana` (no JAX: the validation
subcommands do not need it). Polaris: the project venv on Eagle (`iri/scripts/setup_venv.sh`, `jax[cuda12]`).

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
| V1 muon crossings | `v1compare` | per group (EX, T1, T2, M): crossings/event 1 %; path mean 0.5 %, binned ratio ±2 %; visible-energy mean and MPV 1 %, binned ratio ±3 % (binned ratios allow 2 sigma); t0 bias ≤ 0.1 ns |
| V2-V4 | not written yet | `docs/plan.md` section 5 |

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
  crossings.py  ray-box crossings (JAX if installed, else numpy; identical in float64)
  materials.py  material constants matched to Geant4 11.3.2 (GDML composition/I, Sternheimer, Birks, Urban params)
  eloss.py      restricted MuBetheBloch dE/dx, Urban fluctuation sampler, Birks, delta-ray rate and spectrum
  transport.py  tier-1a muon transport (stage A: straight lines in air)
  tables.py     CrvStep-table readers, primary-muon crossings
  stats.py      robust stats, MPV, binned ratio band
  plots.py      save / write_json / ratio_panel
  analyses/     g4cost, census, transport, v0check, v1compare
scripts/        run_transport_v1.sh (transport the 20 g4ref_v1 shards + V1, one GPVM, ~40 s)
tests/          pytest unit tests (synthetic inputs; geometry tests need only the counters file)
```

## 9. Results so far

| check | sample | result |
|---|---|---|
| V0 | v04 GDML (2026-10-16) | **pass**: 576 bars, max centre residual 3.6e-12 mm |
| census | `g4ref_v1` | run `crvfastsim census`; shares not quoted here |
| V1 | self-test (G4 vs itself) | pass by construction; harness check only |
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

Own CLI for now (`python3 -m crvfastsim.reco vr2|vr2stock|vr1|direct|cost REF.root ...`), to stay out of the shared
`cli.py` while other work is in progress; to be folded in later. Tests: `tests/test_reco_pulsefit.py`.
