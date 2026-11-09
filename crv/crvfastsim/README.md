# crvfastsim — Geant4-free CRV cosmic simulation and its validation

One Python package with a single CLI, for the KPP (extracted) CRV cosmic sample. It follows the conventions of
`crvextrap` (`RobMina/mu2e_track_crv_mcresponse`): definitions live once in shared modules, every question is a
subcommand, hot paths are array code, and batchable analyses use `--emit` / `--merge`. The original plan, goals and thresholds
are in [`../docs/plan.md`](../docs/plan.md) (historical); this file is the living reference for the code.

```
.venv/bin/python3 -m crvfastsim <subcommand> <inputs> [--compare REF ...] [--outdir DIR] [--emit X.json | --merge]
```

`crvfastsim` builds on `mu2efastsim` (the `core/` package of `mu2e-fastsim`: energy loss, materials,
world, generic crossings/transport/tables, statistics, plots, key matching, the IRI job tools). Put both on the path:
from the umbrella root, `export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker` (what `site.env` does), or
`pip install -e core -e crv` in a venv. Needs numpy, pandas, uproot, matplotlib, jax (CPU is enough) and pytest.
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

Persisted coverage, event-paired rate errors, individual-sector and low-momentum diagnostics are available through
`transport-validation`; see `../docs/transport_validation.md` (branch `park/named-run2`). It supplements,
not replaces, the established V1 gates. V1 requires valid fit and populated shape evidence; unsupported checks fail
with explicit reasons. Completed comparisons/merges exit 1 on FAIL after writing results; `--emit` is accumulation
only. Numerical physics tolerances are unchanged; the linked document defines the evidence-validity guards.
`transport-validation` includes event-paired crossing-mean and energy/path errors and matched/unmatched diagnostics.
`transport-paths` decomposes saved matched lengths into entry-ray chords and G4 path excess, retaining all sectors,
clean/nonstopping populations and explicitly truth-conditioned incidence/momentum strata. Neither retunes transport.

## 5. Validation thresholds (agreed 2026-10-15)

| level | subcommand | thresholds |
|---|---|---|
| V0 geometry | `v0check` | every bar centre and half length within 1 um; same ids at every index; no rotations |
| V1 muon crossings | `v1compare` | per group (EX, T1, T2, M): crossings/event 1 %; path mean 0.5 %; visible-energy mean and MPV 1 % (MPV allows 2 sigma of its fit error); path and visible-energy binned ratios chi2/ndf ≤ 1.5 over 20 bins (user, 2026-10-16: the earlier per-bin ±2 %/±3 % after 2 sigma failed identical samples 22 % of the time; the band values are still reported); t0 bias ≤ 0.1 ns |
| V2 response (path A) | `v2compare` | per group: seq-0 pulse PE median 2 % (2 sigma allowed); PE spectrum 5-150 PE in 20 equal-content bins, ±5 % after 2 sigma; coincidence clusters per event 1 % (2 sigma allowed); per-bar rate = events with a ≥ 5 PE pulse in the bar, all 576 bars, chi2/ndf ≤ 1.5 and no bar beyond 5 sigma (user, 2026-10-16: counted once per event and bar, since counting each SiPM's pulse gave ~4 correlated entries per crossing). Per-layer efficiency of truth crossings is a diagnostic only (the agreed observable uses extrapolated tracks and the gap width D) |
| V3 tier 1b (step level) | `tier1b-check` | per group, held-out shards: descendant visible energy per muon crossing (same bar, other bars), non-muon visible energy and bar hits, within 10 % relative (plan V3); descendant positions in the bar (no midpoint on a face) |
| V4 photons | `v4photons` | same CrvSteps into the fast photon stage and into Offline's `MakeCrvPhotons` (PyROOT, `response/stockphotons.py`), same yield deviations: photons per SiPM per event within 0.5 % (paired error, 2 sigma allowed); arrival time minus step start at the 10/50/90 % levels within 0.2 ns, measured as the horizontal distance between the two CDFs at the reference quantile (the tables store delays in whole ns, so a raw quantile can jump a plateau: 0.84 ns at a median on identical laws); `--jax` tests the JAX twin, `--no-stock` compares it with the numpy port without Offline |
| V4 SiPM charges | `v4sipm` | photons from the validated photon port, fed to the fast SiPM stage and to Offline's `MakeCrvSiPMCharges` (PyROOT, `response/stocksipm.py`): PE per SiPM per event within 0.5 % (paired error, 2 sigma allowed); charge time relative to the step start at the 10/50/90 % levels within 0.2 ns (CDF distance); `--jax`, `--no-stock` as `v4photons` |

**Fast response** (`response/`). `lookup.py` reads the v6 photon lookup tables (Mu2e-internal: read at run time from
`$CRVFS_LOOKUP_DIR`, default the Mu2e CVMFS DataFiles area; never committed, never in an image; on Polaris copied into
the Eagle work area for a test and deleted after it). `photons.py` is a vectorised port of `MakeCrvPhotons` and
`CrvPhotonGenerator` with the `epilog_extracted_v05` sector tables, yields and reflectors, `photonYieldScaleFactor` 0.94
and the per-channel `CRVPhoton` deviations (CSV from `dbTool`, `Sim_best v1_5`); steps with a NaN time are skipped as in
the stock module. Photon numbers per step and type are drawn once (Gauss above 10, else Poisson, divided over 0.1 mm
substeps), arrivals as Binomial(n, p) per substep and type (the stock code draws n Bernoullis: same law).
`photons_jax.py` is the JAX twin (two jitted kernels, fixed power-of-two buckets, float64; bit-identical per key).
`bench.py` times the substep kernel on synthetic tables (no Mu2e-internal input).
`sipm.py` ports `MakeCrvSiPMCharges` (production extracted settings: 40 x 40 pixels with the 4 x 4 centre dead, photon
map `photonMap.root`, Vov 3 V, tau 13.3 ns, avalanche p1 0.607 p2 2.7, cross-talk 0.04, ThermalRate 0 from
`digitize/NoField.fcl`, traps 0). It is vectorised across SiPMs: step k handles photon k of every SiPM, then its
cross-talk cascade breadth-first (m hits on one pixel in one generation fire with 1 - (1 - p)^m, the stock serial
order). `sipm_jax.py` is the JAX twin: a `lax.scan` over photon index with 3 cross-talk generations of width 4 (thermal
noise and traps not in the twin).

**Tier 1b library** (`tier1b.py`). Tier 1a writes only the primary muon. Tier 1b adds what G4 also makes: the muon's
descendants (deltas above the production cut, decay electrons) and the non-muon CRY primaries with their showers,
resampled from G4 CrvStep tables of **training** shards and checked on others.
- *Descendants*: each fast crossing draws a G4 crossing from the same (sector group, momentum, path-length) bin; edges
  are per-group quantiles, path length finer in both tails (5, 15, 33, 67, 90, 97 %). Same-bar descendants are offsets from
  the muon entry, rotated to the fast direction; if they fall outside the fast bar they keep the rotated position
  along the bar and take their G4 position across it (thickness, width). Other-bar descendants keep the bar structure
  (G4 bar offset and in-bar position, shifted along the bar with the muon entry); with no bar of the same length axis
  at that offset (module gap, sector edge) the step goes to the nearest such bar in the transverse plane
  (`nearest_bar`). `fit_inside` keeps every midpoint 0.1 mm inside the box. No descendant step is dropped.
- *Non-muon primaries*: each non-muon CRY primary of the event reaches the CRV with the library probability of its
  (pdg, kinetic-energy) bin and then gets a recorded G4 shower of that bin (absolute positions; bar indices of the
  library event, so a geometry move does not move them).
- *Default asset set* (`tier1b_assets.py`, [docs/tier1b-default.md](../docs/tier1b-default.md)): for extracted_v04 the
  callers (`tier1b-apply`, `runchain`, `e2e-bench`, the endpoint kits) use `strict_v5`: the association-certified
  library with step-tree roots, crossing context, endpoint containment, point and topology placement,
  never-visited-layer and near-miss templates from the v3 sidecar, chord placement of the same-bar descendant steps,
  and root-coherent draws (a delta ray and its cascade stay together along the fast muon within a module). The previous
  default `strict_v4` (whole-muon draws of hard library muons) is on branch `park/tier1b-strict-v4`; other option combinations run with `tier1b-apply --library` and explicit placement flags (docs/tier1b-default.md). The named run configurations have no library of their own and need `--library`.
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
| `tier1b-apply` | add library descendants and showers to fast tables (default set `strict_v5`, `--tier1b-dir`, or a custom `--library` with placement flags; `--seed`) | `fast1b.*.root` |
| `tier1b-check` | V3 step-level closure per group vs G4 tables of the same shards (`--compare`) | `tier1b_check.json` |
| `runchain` | the whole fast chain on CRY gen tables (tier 1b: the configuration's set or a custom `--library`; `--calib` CRVCalib + channel-map CSV, `--deviation`, `--jax`, `--max-events`, `--compare` stock reco tables): clusters per generated event per sector type, cost per stage | `runchain_stats.json` |
| `v4sipm` | V4 SiPM-charge stage vs stock `MakeCrvSiPMCharges` on the same photons (`--deviation`, `--max-steps`, `--jax`, `--no-stock`) | `v4sipm_stats.json` |
| `v4photons` | V4 photon stage vs stock `MakeCrvPhotons` on the same steps (`--deviation`, `--max-steps`, `--jax`, `--no-stock`); the stock arm needs `muse setup SimJob ...` plus the `pyenv ana` site-packages on `PYTHONPATH` for pandas | `v4photons_stats.json` |
| `pulsematch` | V2 diagnostic: per-channel PE of four arms (`--fast`, `--fast-mu`, `--ref`, `--ref-mu`) on the same events; the secondaries' added PE per muon-only PE bin, G4 vs fast | `pulsematch_stats.json` |
| `crossingpos` | V2 diagnostic: primary-muon crossing position along and across the bar, fast vs G4, matched by event and bar (`--compare`) | `crossingpos_stats.json`, `crossingpos_along.png` |
| `v2origin` | V2 diagnostic: pulses per PE bin and light yield by deposit origin, two arms (`--steps`, `--compare`, `--compare-steps`, `--generated`) | `v2origin_stats.json` |

**Batch + merge** (same contract as crvextrap §7): workers `--emit batch_NN.json` on a slice of files (one job per
GPVM), then `--merge "batches/*.json"` locally. Merged counts are sums, so the merge equals a single pass.

## Opt-in device transport

`transport`, `runchain` and `e2e-bench` accept `--transport-backend jax` (NumPy remains the default).
The straight-line stages A/C numeric batches run on the selected JAX device; `--tier1b-backend jax` moves tier 1b too.
Unsupported tail/sliced/below-plane variants and unknown materials raise rather than silently changing the model.
This path is experimental, not a speed default. See [`docs/device.md`](../docs/device.md)
for the physics, bounded-shape, RNG, overflow and validation contracts.

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
  chain.py      the whole chain without art: status mask, TDC0 per (event, FEB), digis -> reco waveforms, reco calls
  response/     lookup (v6 tables), photons / photons_jax / stockphotons (photon stage), sipm / sipm_jax / stocksipm
                (SiPM-charge stage), bench
  analyses/     g4cost, census, transport, g4muon, v2origin, pulsematch, crossingpos, secondaries, tier1b, runchain, e2e_bench
  validation/   v0check, v1compare, v2compare, v4photons, v4sipm, transport_validation; DQM histogram compare/coverage,
                pilot, paired_rates, paired_shapes, event_shapes, reco_detail (old paths analyses.<name>, valhist.<name>
                are aliases for one release)
  reco/         fast CRV reconstruction (section 10; subcommands reco-*)
mu2efastsim (core/): eloss, materials, world, crossings, transport, tables, stats, plots, match
scripts/        run_transport_v1.sh (transport the 20 g4ref_v1 shards + V1, one GPVM, ~40 s);
                run_v2_arms.sh (V2 path A arms for one shard, ~1 min per art job)
tests/          pytest unit tests (synthetic inputs; geometry tests need only the counters file)
```

## 9. Results

The validation results, with sample, threshold, date and commit, are in [`../docs/validation.md`](../docs/validation.md);
what is open or not validated is in [`../docs/limitations.md`](../docs/limitations.md).

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
| `reco/compare.py` | V-R1/V-R2 matching and summaries, KS |

Subcommands of the package CLI: `python3 -m crvfastsim reco-vr2|reco-vr2stock|reco-vr2jax|reco-bench|reco-vr1|
reco-cost|reco-inspect REF.root ...` (alias `python3 -m crvfastsim.reco <cmd>`); key joins and the KS test
come from `mu2efastsim.match`. Tests: `tests/test_reco_pulsefit.py`.
