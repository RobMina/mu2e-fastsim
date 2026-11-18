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

## Tasks

`python -m benchmarks.fastsim_score.tasks` prints this table (`tasks.py`). Each task is a directory of the reference tree.

| task | what | boundary | stock reference | scored | floor | rate |
|---|---|---|---|---|---|---|
| W1 | Muon beam | TS-entrance mu/pi -> stopping-target stops | Geant4 TS + DS stages, expectation over stock re-seeds | stop foil, time, radius, DS-entrance momentum (paired per shard); stops per input (measured) | none yet (threshold 2) | TS-entrance mu/pi per s |
| W2 | Conversion electrons | CE stops -> CE track observables | full Geant4 events -> stock digi + LoopHelix reco | ValKalSeed histograms, paired shape per sub-shard (yields per generated electron are a separate study) | none yet (threshold 2) | events/s |
| W3 | DIO electrons | DIO births -> CE track observables | Geant4 tracker steps of the same events -> stock digi + reco | CE-selected momentum spectrum 95-106 MeV/c; tail counts above 95 and 100 MeV/c and all CE-selected | none yet (threshold 2) | generated DIO muons per s |
| W4 | RPC electrons | RPC births -> tracker entry | Geant4 RPC sample at the tracker entrance | tracker-entry fiducial yield (no stock reco reference after the response yet) | none yet (threshold 2) | generated RPC captures per s |
| W5 | CRV cosmics | CRY cosmics -> CRV coincidence clusters | Geant4 steps -> stock CRV response + reco | ValCrv histograms, paired shape per shard; reco pulse yield | stock with another seed | events/s |
| W6 | Pileup mixing | primary steps + donor pools -> mixed tracker digis + CRV pulses | stock mixing of the same primaries | digis per plane, pulses per sector (paired by event); tracker and CRV totals | none yet (threshold 2) | events/s |
| W7 | Calorimeter | CE calorimeter steps -> calo digis, hits, clusters | Geant4 calorimeter steps -> stock calo digi + reco | cluster E, t, R, N (paired shape per block); cluster yield | stock with another seed | events/s |
| W8 | Tracker trigger | StrawDigis -> tracker trigger path decisions | the stock trigger on the same digis | accept fraction per path (measured, paired error, tolerance 2 % of the acceptance) | none yet (threshold 2) | events/s |
| W9 | CRV design model | CRV module design parameters -> veto (in)efficiency | full-simulation CRV veto comparison of the same design (stock arm of the design study) | inefficiency per momentum bin; efficiency per CRV sector | none yet (threshold 2) | events/s |
| CRVLY | CRV light-yield (lookup) table generator | CRV counter geometry and optics -> photon arrival lookup tables | the v6_0 lookup tables (2019 standalone CRV optical simulation; the generator's default v6 model) | bin-by-bin arrival probability per table type (scintillation, Cerenkov in scintillator, Cerenkov in fibre), error binomial on the photons generated per bin; grid jobs and cost per table | none yet (threshold 2) | tables/s |
| PTGT | Production-target chain | protons on target -> particles at the transport-solenoid entrance | G4beamline on the same target configurations (the design board's rods) | stopped muons per POT and mu- + pi- per POT at the TS entrance per configuration (measured yields); p, cos theta, r and t at the TS entrance (shape); throughput per POT | replicate G4beamline runs of the same configuration | POT/s |

## Coverage on the existing validation outputs (2026-11-18)

Scored with `--allow-missing` from the validation arms each workflow already delivered and their recorded bench rows;
no new stock or fast jobs. The reference trees are kept privately. The arms are **not all on current main**: each row
names the commit of its fast arm and, where different, of its throughput row.

| task | dtype | validity | flagged / histograms | yields passing | throughput x | device | fast-arm commit (rate commit) |
|---|---|---|---|---|---|---|---|
| W1 | float32 | 1.000 | 0 / 4 | 1 / 1 | 956 | A100-80 | a282de98 |
| W1 | float64 | 1.000 | 0 / 4 | 1 / 1 | 612 | A100-80 | a282de98 |
| W2 | float32 | 0.889 | 1 / 9 | 0 / 0 | 361 | A100-80 | 1845918 (8064739) |
| W2 | float64 | 0.889 | 1 / 9 | 0 / 0 | 274 | A100-80 | 1845918 (8064739) |
| W3 | float32 | 1.000 | 0 / 1 | 3 / 3 | 163 | RTX PRO 6000 MIG 1g | 6b1f2c2 |
| W3 | float64 | 1.000 | 0 / 1 | 3 / 3 | 60 | RTX PRO 6000 MIG 1g | 6b1f2c2 |
| W4 | float32 | 1.000 | 0 / 0 | 1 / 1 | 81 | RTX PRO 6000 MIG 1g | 6b1f2c2 |
| W4 | float64 | 1.000 | 0 / 0 | 1 / 1 | 34 | RTX PRO 6000 MIG 1g | 6b1f2c2 |
| W5 | float32 | 1.000 | 0 / 12 | 1 / 1 | 1310 | A100-80 | 862eb05 (c8ce9a1c) |
| W5 | float64 | 1.000 | 0 / 12 | 1 / 1 | 1091 | A100-80 | 862eb05 (c8ce9a1c) |
| W6 | float32 | 1.000 | 0 / 2 | 2 / 2 | 108 | A100-80 | a05e15e |
| W6 | float64 | 1.000 | 0 / 2 | 2 / 2 | 99 | A100-80 | a05e15e |
| W7 | float32 | 0.750 | 1 / 4 | 1 / 1 | 21 | A100-80 | 862eb05 (#86 a542786) |
| W7 | float64 | 0.750 | 1 / 4 | 1 / 1 | 15 | A100-80 | 862eb05 (#86 d56250e) |
| W8 | float32 | 1.000 | 0 / 0 | 20 / 20 | 44 | A100-80 | 390ea7d |
| W8 | float64 | 1.000 | 0 / 0 | 20 / 20 | 38 | A100-80 | 390ea7d |
| W9 | float32 | not scored: no float32 arm (both dtypes are required) | | | | | |
| W9 | float64 | 0.500 | 1 / 2 | 1 / 1 | not scored |  | 67bfc9cd |

Mean validity 0.928 over the 17 scored arms; throughput geometric mean x135 over the 16 with a rate.

- **W1:** stock is the expectation over stock re-seeds; values with the paired error per bin. Throughput is against a stock proxy.
- **W2:** paired shape per sub-shard (70); the per-event yield of this sample is conditioned, so W2 is scored on shapes only (NSeed flags). No floor (threshold 2).
- **W3:** one spectrum and three tail counts on the same generated muons; old umbrella code and a stock proxy (the CE rate). The 3-sigma yield rule passes the above-100 MeV/c tail (1.11 / 1.14), which a 2-sigma rule would not.
- **W4:** tracker entry only: there is no stock RPC reconstruction to compare after the response.
- **W6:** the float32 arm is float32 CRV with a float64 tracker; there is no pure float32 arm.
- **W7:** the validated arm is the waveform chain, and so is its rate row; the parametric default on main has no same-events stock comparison yet. The cluster-radius flag is the inner edge.
- **W8:** per-path accept fractions with the thread's 2 % tolerance; under this scorer's rule (3 sigma + tolerance) all 20 paths pass, where the thread's own rule (2 % or 2 sigma) passes 19.
- **W9:** float64 only (both dtypes exist only for an earlier model definition); no floor and no warm rate at a stock-slot boundary, so validity only. The sector efficiencies flag, driven by sectors with few events.
- **CRVLY, PTGT:** defined, not scored: their references wait for the generator's full tables and a measured stock cost per table, and for the device arm on every configuration with a settled response bank.
- Missing everywhere: a reseeded-stock floor for W1, W2, W3, W4, W6, W8 and W9, and bench reports from current main for most workflows.

## Layout

```
submission/<W>/<dtype>/hists.json   fast arm: {"hists": {key: {"edges", "counts" | "shards"}}, "yields": {...}}
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
```

`build_bundle` packages the scorer as a Codabench results-submission bundle (stock `codalab-legacy:py312` image).
With a real reference, it is for an unpublished competition or local scoring only.

Tests: `tests/test_benchmarks_fastsim.py`.
