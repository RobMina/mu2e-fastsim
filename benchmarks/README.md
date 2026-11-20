# Benchmarks: the three metrics of a fastsim commit

`benchmarks/fastsim_score` scores one commit of this repository on the three metrics every fast workflow reports.
Run on any commit, it gives a score that compares with every other commit's:

| Metric | How it is computed | Ranking |
|---|---|---|
| **Validity** | per stock histogram, chi2/ndf of the fast arm against stock on the same generated events, flagged when chi2/ndf > max(2, 3 x floor) and p < 1e-3 (floor = stock with another seed against stock); yields pass when \|ratio - 1\| - 3 se <= tolerance. Validity = (1 - flagged fraction) x (fraction of yields that pass) | key: mean over workflows and both dtypes |
| **Throughput multiplier** | `mu2efastsim.bench` warm rate / stock events per second per single-core grid slot at the same boundary | geometric mean; self-reported, labelled with the device |
| **SU per event** | SU per hour of the device / (3600 x warm rate) | private scoring only (`--private`), never on a leaderboard |

Both float32 and float64 are required for every workflow. The statistics are the ones the validation code here
already uses:
- the two-sample sum `(n1 - n2)^2 / (n1 + n2)` on the same events (`crvfastsim.validation.compare`);
- a scaled shape test (mode `shape`);
- the paired per-shard test when both arms carry shards (`trkfastsim.validation.chain_pair`).

The scorer uses NumPy only and imports nothing from this repository, so a Codabench worker can run it unchanged.

## Histogram kinds, modes and yields

| Kind | hists.json entry | Statistic |
|---|---|---|
| counts | `edges` + `counts`, or `edges` + `shards` | mode `same_events` (independent Poisson difference), `shape` (scaled), or paired per shard; with `shape` and shards, the **paired shape** test (candidate scaled to the reference total, chi2/ndf x (S-3)/(S-1), p from the raw sum), which is the Run 1A DQMVAL comparison exactly |
| values | `edges` + `values` + `errors` | sum (v1 - v2)^2 / (e1^2 + e2^2): tables and derived quantities with their own per-bin error, e.g. a lookup table, a paired comparison whose paired error sits on the reference side |
| efficiency | `edges` + `passed` + `total` | per-bin binomial difference |

A yield is a count, per-shard counts, or a measured `{"value", "error"}` (stops per POT, an accept fraction). A task
may be yields only (per-path accept fractions); its validity is then the fraction of yields that pass. A task whose
`reference.json` has `"stock_rate_per_slot": null` is scored on validity only. `rate_unit` names what the rate counts
(events, POT, tables); a bench report in another unit is an error. `stock_jobs_per_unit` and a bench `jobs_per_unit`
add the grid jobs replaced per unit of work. `--allow-missing` (coverage runs only, never a leaderboard) lists a
missing or unscorable arm instead of failing.

### Extensions for the B1-B4 cards (optional keys of `reference.json`)

| Key | Arm files | What the scorer does |
|---|---|---|
| `yield_max_se` `{name: se}` | - | a yield with a larger se fails as underpowered, so a small sample cannot pass a tight tolerance (B1's 0.2 % peak position needs se <= 0.067 %) |
| `classifier` `{"file", "seed"}` | `features.npz` (array `X`, one row per object) in `stock/`, `floor/` and each arm | fixed NumPy logistic regression on standardised quadratic features, 50/50 train/test, AUC on the test half; `cls_score = clip(2 (1 - AUC), 0, 1)`, 1 = indistinguishable; the floor's score is the ceiling. Reported beside validity, not in it (B1, B2) |
| `track_agreement` `{"file", "threshold_sigma", "min_fraction"}` | `tracks.npz`: stock `id`, `params`, `sigma` (+ `truth`, `good`); arm `id`, `params` (+ `sigma`, `good`) | agreement = fraction of stock tracks whose fast fit is within threshold x sigma_stock in every parameter (unmatched tracks disagree); the paired good-fit ratio is a yield; with `min_fraction` the agreement is a pass/fail yield; pulls against truth reported (B3). A tracks-only task needs no hists.json |
| `scaling_batches` `[10, 1000, 100000]` | bench.json `scaling: [{"batch", "rate"}]` | multiplier per batch size and the crossover batch (first multiplier > 1); headline rate = `warm_rate`, else the largest batch (B3) |
| `group` + `part` (`dev`, `place`, `design`) | - | per group and dtype: portability ratio = mean place validity / dev validity, adaptation score = mean design validity; one task per layout (B2) |
| `capture_budget` | bench.json `capture_entries` | an arm over its adaptation budget, or not declaring it, is an error (B2 design moves) |
| `peak_fit` `{name: {"start_above", "start_width", "n_boot", "seed"}}` | hists.json `peaks: {name: counts or shards}` | the peak position in each arm by CAL1's Gaussian core (a Gaussian fitted to the bins, weights 1/n, window [mu - 1.5 s, mu + 2 s], six iterations); the ratio is a yield, error by a fixed-seed paired bootstrap over the shards (200 replicates); `yield_max_se` applies to it, and the arm reports `power_ok` (B1's CE peak) |
| `yield_tolerances` `{name: tol}` | - | per-yield tolerance instead of `yield_tolerance` |
| `max_memory_mb_per_thread` | bench.json `memory_mb_per_thread` | `memory_ok`, reported (B1: 250 MB) |
| - | bench.json `"host": "mu2eg4"` + `baseline_rate` | a model inside Geant4: multiplier = rate / its own model-off rate on the same node and primaries (B1 Mu2eG4 host) |

The result also carries `cls_score`, the mean classifier score over the arms that have one.

## Tasks

`python -m benchmarks.fastsim_score.tasks` prints this table (`tasks.py`). Each task is a directory of the reference tree.

| task | what | boundary | stock reference | scored | floor | rate | status |
|---|---|---|---|---|---|---|---|
| W1 | Muon beam | TS-entrance mu/pi -> stopping-target stops | Geant4 TS + DS stages, expectation over stock re-seeds | stop foil, time, radius, DS-entrance momentum (paired per shard); stops per shard | one stock re-seed | TS-entrance mu/pi per s | scored |
| W2 | Conversion electrons | CE stops -> CE track observables | full Geant4 events -> stock digi + LoopHelix reco | ValKalSeed histograms, paired shape per sub-shard (yields per generated electron are a separate study) | none yet (threshold 2) | events/s | scored |
| W3 | DIO electrons | DIO births -> CE track observables | full Geant4 events (every tracker step of the DIO dts) -> stock OffSpill digi + LoopHelix reco | CE-selected momentum spectrum 95-106 MeV/c; tail counts above 95 and 100 MeV/c and all CE-selected | none yet (threshold 2) | generated DIO muons per s | scored |
| W4 | RPC electrons | RPC births -> tracker entry | Geant4 RPC sample at the tracker entrance | tracker-entry fiducial yield (no stock reco reference after the response yet) | none yet (threshold 2) | generated RPC captures per s | scored |
| W5 | CRV cosmics | CRY cosmics -> CRV coincidence clusters | Geant4 steps -> stock CRV response + reco | ValCrv histograms, paired shape per shard; reco pulse yield | stock with another seed | events/s | scored |
| W6 | Pileup mixing | primary steps + donor pools -> mixed tracker digis + CRV pulses | stock mixing of the same primaries | digis per plane, pulses per sector (paired by event); tracker and CRV totals | none yet (threshold 2) | events/s | scored |
| W7 | Calorimeter | CE calorimeter steps -> calo digis, hits, clusters | Geant4 calorimeter steps -> stock calo digi + reco | cluster E, t, R, N (paired shape per block); cluster yield | stock with another seed | events/s | scored |
| W8 | Tracker trigger | StrawDigis -> tracker trigger path decisions | the stock trigger on the same digis | accept fraction per path (measured, paired error, tolerance 2 % of the acceptance) | none yet (threshold 2) | events/s | scored |
| W9 | CRV design model | CRV module design parameters -> veto (in)efficiency | full-simulation CRV veto comparison of the same design (stock arm of the design study) | inefficiency per momentum bin; efficiency per CRV sector | none yet (threshold 2) | selected CE-proxy events per s | scored |
| CRVLY | CRV light-yield (lookup) table generator | CRV counter geometry and optics -> photon arrival lookup tables | the v6_0 lookup tables (2019 standalone CRV optical simulation; the generator's default v6 model) | bin-by-bin arrival probability per table type (scintillation, Cerenkov in scintillator, Cerenkov in fibre), error binomial on the photons generated per bin; grid jobs and cost per table | none yet (threshold 2) | tables/s | defined |
| PTGT | Production-target chain | protons on target -> particles at the transport-solenoid entrance | G4beamline on the same target configurations (the design board's rods) | stopped muons per POT and mu- + pi- per POT at the TS entrance per configuration (measured yields); p, cos theta, r and t at the TS entrance (shape); throughput per POT | replicate G4beamline runs of the same configuration | POT/s | scored |
| B1 | Calorimeter shower model inside and outside Geant4 (FS-CALO-G4) | e+-/gamma state at a calorimeter disk front -> crystal deposits (stock calo digi + reco downstream) | Mu2eG4 with the model off on the same CeEndpoint primaries, one Offline build | cluster E, time, x, y, radius, size, clusters per event, hit E; yields: events with a cluster > 10 MeV, CE peak position (tolerance 0.2 %, se <= 0.067 %); classifier score on per-cluster features (reported beside validity) | model off with another seed | events/s | defined |
| B2 | Geometry-portable CRV module interaction (FS-CRV-MOD) | entry state at a CRV module envelope (module frame) -> the module's CrvSteps + exiting particles | full Geant4 per layout: dev = held-out nominal-layout shards; place = placement moves of the current module type; design = the CRV design study's short-listed layouts (stock from its grid production, reused) | per layout: DQMVAL CRV histograms (steps, pulses, coincidences), exit-particle yield and spectrum; portability ratio = mean place validity / dev validity; adaptation score = mean design validity at the declared capture budget; classifier score on per-coincidence features | stock with another seed, per layout | events/s | defined |
| B3 | Batched track fits against KinKal (FS-TRKFIT) | frozen stock reco inputs per track (combo hits, time cluster, seed) -> fit parameters and covariance at the tracker entrance | stock KinKal on CPU on the same inputs (agreement 1 by construction; sets the throughput basis); KinematicLine (extracted), CentralHelix cosmics, LoopHelix CE | agreement: fraction of tracks within 0.1 sigma of the stock fit in every parameter (ranking key); good-fit ratio fast/stock (paired); pulls against MC truth; throughput scaling curve at batch 10, 10^3, 10^5 tracks with the crossover batch | none yet (threshold 2) | tracks/s | defined |
| B4 | GPU e+-/gamma transport for the flash (FS-FLASH-GPU) | flash primaries at one stage boundary fixed before any run (recommended: DS entrance) -> tracker StrawGasSteps, the DS/tracker regions offloaded through G4VTrackingManager | Mu2eG4 without offload on the same primaries and build; CPU controls for the stock behaviour the GPU cannot copy (no low-energy kill; tracks killed on leaving the DS) | StrawGasStep edep, multiplicity, time and per-plane histograms; the flash_edep objective as a yield (tolerance proposed 3 %) | stock with another seed | events/s | defined |

## Coverage (final rescore of the sprint, 2026-11-19)

Every fast arm runs with its workflow's own validation recipe, on the same events as its frozen stock reference, and
the bench timer uses the published headline pass sizes. No stock jobs were run.
- **Commits:** the first scoring was the full rescore on main `c03dbd13`. The final rescore then reran only the arms
  whose code changed, on merged main. The **commit** column says which commit each row was scored on.
- **PTGT** was scored on #293's head before it merged. Its chain code (`ptgt_chain_jax`, `tsdevice`, the slab and
  G4beamline readers) is identical to main `32eea930`. The merge only renamed one method and completed the bank
  manifest builder.
- **Scores:** from `score --allow-missing` (coverage) over 20 of 20 arms (W1-W9 and PTGT): **mean validity 0.900**
  and a **throughput geometric mean of x1,012**.
- The last column is the same task scored on the older arms. The reference trees are kept privately.

| task | dtype | commit | validity | flagged / histograms | yields passing | throughput x | device | before (scored 2026-11-18 on older arms) |
|---|---|---|---|---|---|---|---|---|
| W1 | float32 | c03dbd13 | 1.000 | 0 / 4 | 1 / 1 | 949 | A100-80 | 1.000, x956 |
| W1 | float64 | c03dbd13 | 1.000 | 0 / 4 | 1 / 1 | 613 | A100-80 | 1.000, x612 |
| W2 | float32 | c03dbd13 | 0.889 | 1 / 9 | 0 / 0 | 467 | A100-80 | 0.889, x361 |
| W2 | float64 | c03dbd13 | 0.889 | 1 / 9 | 0 / 0 | 306 | A100-80 | 0.889, x274 |
| W3 | float32 | c03dbd13 | 1.000 | 0 / 1 | 3 / 3 | 169 | A100-80 | 1.000, x163 (primary-only stock arm) |
| W3 | float64 | c03dbd13 | 1.000 | 0 / 1 | 3 / 3 | 115 | A100-80 | 1.000, x60 (primary-only stock arm) |
| W4 | float32 | c03dbd13 | 1.000 | 0 / 0 | 1 / 1 | 96 | A100-80 | 1.000, x81 |
| W4 | float64 | c03dbd13 | 1.000 | 0 / 0 | 1 / 1 | 73 | A100-80 | 1.000, x34 |
| W5 | float32 | c03dbd13 | 1.000 | 0 / 12 | 1 / 1 | 1299 | A100-80 | 1.000, x1310 |
| W5 | float64 | c03dbd13 | 1.000 | 0 / 12 | 1 / 1 | 1090 | A100-80 | 1.000, x1091 |
| W6 | float32 | c03dbd13 | 1.000 | 0 / 2 | 2 / 2 | 191 | A100-80 | 1.000, x108 |
| W6 | float64 | c03dbd13 | 1.000 | 0 / 2 | 2 / 2 | 180 | A100-80 | 1.000, x99 |
| W7 | float32 | c03dbd13 | 0.750 | 1 / 4 | 1 / 1 | 2582 | A100-80 | 0.750, x21 |
| W7 | float64 | c03dbd13 | 0.750 | 1 / 4 | 1 / 1 | 2315 | A100-80 | 0.750, x15 |
| W8 | float32 | 213edff8 | 1.000 | 0 / 0 | 22 / 22 | 350 | A100-80 | 1.000, x349 (c03dbd13, before #283) |
| W8 | float64 | 213edff8 | 1.000 | 0 / 0 | 22 / 22 | 320 | A100-80 | 1.000, x305 (c03dbd13, before #283) |
| W9 | float32 | c03dbd13 | 0.500 | 1 / 2 | 1 / 1 | 38635 | CPU, 4 cores | not scored |
| W9 | float64 | c03dbd13 | 0.500 | 1 / 2 | 1 / 1 | 38499 | CPU, 4 cores | 0.500 |
| PTGT | float32 | 32eea930 | 0.864 | 0 / 4 | 57 / 66 | 58341 | A100 | 0.561, x88132 (absorbing TS-window rings) |
| PTGT | float64 | 32eea930 | 0.864 | 0 / 4 | 57 / 66 | 39112 | A100 | 0.561, x53064 (absorbing TS-window rings) |

- **W1:**
  - the stock arm is the expectation over 16 stock re-seeds, now compared per shard (paired, yield included);
  - the floor is one stock re-seed;
  - throughput is against a stock proxy.
- **W2:** paired shape per sub-shard (70); yields are not scored from this sample (per-event yields are conditioned);
  the NSeed flag remains.
- **W3: the stock arm is the full-event stock reconstruction** (rebuilt 2026-11-19; the fast arms are unchanged).
  - The first rescore scored 0 because the frozen stock arm digitised only the DIO electron's own Geant4 steps. Main's
    default reco response (the full-event response, the default since #149) models the reconstruction of full events,
    whose CE-selected yield is 0.93 of that primary-only arm. The 0.926 was this mismatch, not a fidelity loss.
  - The stock arm is now every tracker step of the same DIO dts through the stock OffSpill digi + LoopHelix reco, on
    the same 200 held-out shards. Against it, the fast arm gives CE-selected 0.994 / 0.993 (float32 / float64), above
    95 MeV/c 0.999 / 0.997, above 100 MeV/c 1.09 / 1.08 +- 0.05; the spectrum's chi2/ndf is 0.88 / 0.94.
  - The older arm (CE-only response) scored 1.000 only against the primary-only arm; it is about 1.08 of the full-event
    stock arm, so its "before" entry is not comparable with this row.
- **W4:** the tracker-entry stage only; there is no stock RPC reconstruction after the response.
- **W5, W6, W7:**
  - W6's float32 arm is now pure float32. W6 runs about 1.8x faster on main than on the older arm.
  - W7 is the parametric default: x2,582 / x2,315, with the cluster-radius flag at the inner edge as before.
- **W8:** main as is (device loops on); 22 tested paths. Under this scorer's rule (3 sigma + 2 %) all of them pass.
- **W9:**
  - float32 and float64 on the same paired events;
  - throughput is against the stock Geant4 + CRV reconstruction core-seconds per selected event (existing shard
    timings, one CPU core), with the model timed from the CRV-entry states;
  - the sector efficiencies flag (sectors with few events).
- **Floors:** none for W2, W3, W4, W6, W8 or W9; making one would need new stock jobs.
- **PTGT:** 33 design-board rods, two seeds of 1M POT each (pooled), both dtypes.
  - The TS-window steel rings are crossed as thin steel slabs. The first freeze treated them as perfect absorbers, which
    gave mu- + pi- at the TS entrance 0.93 of G4beamline and validity 0.561 (the "before" column).
  - Stops per POT pass on all 33 rods (median 1.02 / 1.00, float32 / float64). The stock stops error is
    max(spread over the G4beamline shards, Poisson), since the shard spread falls below Poisson on most rods.
  - mu- + pi- per POT at the TS entrance passes on 24 of 33 (median 0.96); the nine failing rods are at 0.94-0.96.
  - The TS-entrance shapes at the nominal rod pass, against G4beamline's replicate run as the floor.
  - Throughput is per POT for the target + PS stage. The slab crossing costs about a third of the absorbing version's
    rate.
- **CRVLY:** defined, not yet scored; its reference waits for full generated tables with a measured stock cost per table.
- **W8:** scored on main `213edff8` with the exact-hits trigger code (#283); all 22 tested paths pass in both dtypes.

## B1-B4: the Geant4 fast-simulation and GPU cards

B1 FS-CALO-G4, B2 FS-CRV-MOD, B3 FS-TRKFIT and B4 FS-FLASH-GPU are tasks of this scorer, with a workflow key per task
(B2 one per layout, B3 one per configuration), the same `hists.json` / `bench.json` / `reference.json` layout, both
dtypes, and references kept privately. The task table gives
each card's boundary, stock arm, metrics, floor and rules; the extensions above are the scorer side. B1 has a
dev reference (below); B2-B4 are defined, not scored.

| Card | Ranking | Beside validity | Reference rule |
|---|---|---|---|
| B1 | validity (cluster histograms; cluster yield; CE peak position at 0.2 % with se <= 0.067 %) | classifier score; throughput on the declared host; memory increase <= 250 MB per thread | beats Gflash profiles and all-energy-in-the-entry-crystal on validity and classifier score |
| B2 | final validity (place + design layouts) | portability ratio, adaptation score at the capture budget, classifier score, exit-particle closure | the absolute-placement hybrid passes dev and fails place; the nominal library unadapted fails design |
| B3 | agreement with stock KinKal within 0.1 sigma | good-fit ratio, pulls, throughput scaling curve and crossover batch | a least-squares helix without material or drift fails agreement |
| B4 | validity (StrawGasStep histograms; flash_edep yield) | throughput (job wall, CPU-only vs offload, same node); engine selected by score | the earlier direct-electron sketch fails |

### B1 card (FS-CALO-G4, task `B1`)

One calorimeter shower model, two hosts: a Geant4 fast-simulation model inside Mu2eG4, or the fastsim device chain.
Calorimeter showers are about half of the Geant4 time of a conversion-electron event (measured by killing every
track at the calorimeter entry), so a model that holds the stock calorimeter observables is worth up to 2x in Geant4.
B1 scores any such model against stock with one card, whichever library or network it uses. Format: the five
elements of the MLCommons Science benchmark ontology, motifs, a rubric self-score (`rubric_b1.yaml`), and the
Codabench results-submission bundle of this directory.

**Motifs.** Scientific: High-Energy Physics. AI/ML: Generative. Computing: Throughput Bound, Memory Bound.

**Rubric self-score** (`rubric_b1.yaml`, benchmark-builder `score_benchmark.py`, after the dev reference): software
4, specification 5, dataset 3, metrics 5, reference 2, documentation 4; **mean 3.83** (below the 4.5 endorsement:
Mu2e-internal data, no Mu2eG4-host reference, weak baselines not run, no paper).

**A. Problem specification and constraints.** Given the state of an e+-, e- or photon at the front of a calorimeter
disk (position and direction in the disk frame, energy, time, species), produce the crystal deposits (crystal,
energy, Birks energy, time, longitudinal position) of its shower, which the stock calorimeter digitisation and
reconstruction consume unchanged. A submission names its host (`bench.json` `host`: `mu2eg4` or `device`) and runs
float32 and float64. Fixed constraints: the Offline release and envset named per result; Geant4 11.3.2; a
single-thread CPU job for the Mu2eG4 host; resident-memory increase **<= 250 MB per thread**
(`max_memory_mb_per_thread`, reported as `memory_ok`); entries the model leaves to Geant4 near the disk edges (the
edge rule) are part of the submission and reported (`edge_rule`).

**B. Dataset.** Primaries of the production `CeEndpoint` MDC2025ba files of run 1430; both arms are simulated from
the same primaries (stock = model off, floor = model off with another seed).

| Split | Files | Use |
|---|---|---|
| train | `001430_000000{01,101,...,901}` | building a library or training a model |
| dev | `001430_00001000, 1100, 1200, 1300, 1400` | public practice; results may be iterated on |
| final | `001430_00001500 ... 1900` plus a fresh ~200,000-event CeEndpoint set on reserved seeds | scored once per submission version |

The final split is larger because of the power rule (C). Mu2e simulation data is collaboration-internal (FAIR:
findable, interoperable, not accessible); the public variant is the synthetic `B1` example of `synthetic.py`.

**C. Metrics** (definitions fixed in `calo.py`, statistics in `score.py` and `classifier.py`).
- **Validity** (ranking key). Histograms in 100-event blocks, paired shape test per block, DQMVAL flag rule: cluster
  energy (every cluster); time, x, y, radius and size of clusters above 10 MeV; clusters above 10 MeV per event;
  reconstructed-hit energy. Yields: events with a cluster above 10 MeV, and the **CE peak**, the ratio of the
  leading-cluster peak positions (CAL1's Gaussian core, 1 MeV bins 40-115 MeV), both with tolerance **0.2 %**
  (|ratio - 1| - 3 se <= 0.002). **Power rule:** the CE peak passes only if its standard error (paired bootstrap
  over the event blocks) is also **<= 0.067 %** (tolerance / 3), about 200,000 paired CE events; `power_ok` reports it.
- **Classifier score** (after CaloChallenge, arXiv:2410.21611): per cluster above 10 MeV the vector (energy, time, x,
  y, size, seed-crystal energy fraction, mxx, myy, mxy: energy-weighted second moments of the member crystals),
  stock against fast, a fixed-seed logistic regression on quadratic features, 50/50 train/test of equal samples;
  `cls_score = clip(2 (1 - AUC), 0, 1)` (1 = indistinguishable). Reported next to validity.
- **Throughput multiplier.** Mu2eG4 host: Mu2eG4 time per event model off / model on, same node and primaries
  (`baseline_rate`). Device host: the bench warm rate over the stock rate per grid slot at the same boundary.
- **SU per event:** private scoring only.

**D. Reference solution and baselines.** Reference: the fastsim device chain (`calo/calofastsim/device_chain.py`)
at the first forward direct-CE disk-front entry of each event, with the CAL2 whole-shower library (K8 neighbours,
crystal placement), as validated by W7-CAL; arm driver `calo_device.py`. The Mu2eG4 host has no reference run yet.
Weak baselines (defined, not implemented): Gflash longitudinal and radial profiles with CsI parameters, and all
energy in the entry crystal. Floor: stock with another seed (not produced yet; the flag threshold is then 2).

**Reference on the dev split (2026-11-19).** Device chain at main plus this card's opt-in cluster shape outputs,
CAL2 library, 12,195 paired events (every dev event with a disk-front entry, as W7-CAL), 100-event blocks, CPU,
draw and placement op by op (`calo_device.py --draw eager`), throughput not scored in this run, no floor yet
(flag threshold 2):

| response | dtype | validity | flagged (chi2/ndf) | events > 10 MeV | CE peak ratio | power | cls_score (AUC) |
|---|---|---|---|---|---|---|---|
| parametric (default) | float32 | 0.31 | radius 6.1, size 5.8, hit E 2.7 | 1.0008 +- 0.0029 | 0.9985 +- 0.0030 | fail | 0.86 (0.572) |
| parametric (default) | float64 | 0.31 | radius 6.3, size 5.4, hit E 3.0 | 1.0010 +- 0.0028 | 0.9955 +- 0.0030 | fail | 0.84 (0.582) |
| waveform | float32 | 0.38 | radius 6.2, size 4.5 | 1.0005 +- 0.0028 | 0.9997 +- 0.0033 | fail | 0.87 (0.564) |
| waveform | float64 | 0.38 | radius 6.3, size 5.0 | 1.0001 +- 0.0028 | 0.9996 +- 0.0029 | fail | 0.87 (0.567) |

Both yields are within the 0.2 % tolerance at 3 se, but the CE peak error (0.3 %) is far above the 0.067 % the
power rule asks for, so the peak fails by construction on dev: the dev split is too small to certify the
tolerance (the final split is sized for it). Cluster energy, time, x, y and clusters per event pass. The radius flag
is W7-CAL's inner-edge miss (front-only library; the per-entrant library with an inner-edge face addresses it); the
cluster-size flag is new with this card; the parametric response also flags the hit energy. The weak baselines
are not run yet, so the card cannot yet say the reference beats them.

**E. Documentation and reproduction.**

```bash
python -m benchmarks.fastsim_score.calo_device SUB/B1/float64 --tables EXPORT.root ... --stock CALO_NTUPLE.root ... \
    --library library.npz --cal1-inputs DIR --dtype float64 [--response parametric]     # and float32
python -m benchmarks.fastsim_score.calo ntuple REF/B1/stock CALO_NTUPLE.root ... --events SUB/B1/float64/events.npy
python -m benchmarks.fastsim_score.calo reference REF/B1 [--stock-rate R]
python -m benchmarks.fastsim_score.score SUB REF [--out scores.json]
python -m benchmarks.codabench.build_bundle OUT --reference REF --solution SUB
```

The exports are `CaloLibraryDumper` primary-mode dumps of the DTS files; the stock ntuples are the stock calorimeter
chain's `prd` ntuples of the same files. Every `bench.json` names the Offline commit, envset, library hash and image.

## Layout

```
submission/<W>/<dtype>/hists.json   fast arm: {"hists": {key: {"edges", "counts" | "shards"}}, "yields": {...},
                                    "peaks": {...}}
submission/<W>/<dtype>/features.npz per-object features "X" (tasks with a classifier)
submission/<W>/<dtype>/bench.json   the mu2efastsim.bench report (warm_rate, device, dtype, boundary)
reference/<W>/stock/hists.json      stock arm on the same generated events
reference/<W>/floor/hists.json      stock with another seed (optional)
reference/<W>/reference.json        stock_rate_per_slot, boundary, mode, yield_tolerance, min_count, su_per_hour
```

Reference trees hold Mu2e simulation results and are **never committed**: keep them outside the repository and
point the scorer at them. `synthetic.py` writes a toy example of the whole layout.

## Commands

```bash
python -m benchmarks.fastsim_score.from_root fast_W5_f64.json shard_*.root --select '^Validation/Crv' --yield coincidences=Validation/CrvCoinc/N
python -m benchmarks.fastsim_score.score SUBMISSION REFERENCE [--private] [--out scores.json]
python -m benchmarks.fastsim_score.synthetic EXAMPLE && python -m benchmarks.fastsim_score.score EXAMPLE/submissions/good EXAMPLE/reference
python -m benchmarks.codabench.build_bundle OUT [--reference DIR --solution DIR]
python -m benchmarks.fastsim_score.calo ntuple OUT/B1/stock CALO_NTUPLE.root ... --events events.npy   # B1 arm from stock calo ntuples
python -m benchmarks.fastsim_score.calo_device OUT/B1/float64 --tables EXPORT.root ... --stock CALO_NTUPLE.root ... --library library.npz --cal1-inputs DIR --dtype float64   # B1 device-host arm
python -m benchmarks.fastsim_score.calo reference OUT/B1 [--stock-rate R]                              # B1 reference.json
```

`build_bundle` packages the scorer as a Codabench results-submission bundle (stock `codalab-legacy:py312` image).
With a real reference, it is for an unpublished competition or local scoring only.

Tests: `tests/test_benchmarks_fastsim.py`.
