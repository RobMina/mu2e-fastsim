# Baseline profile of the stock extracted cosmic chain

The reference numbers that the fast simulation is measured against. All tools here are rerunnable.

| file | what it does |
|---|---|
| `fcl/prof_sim_extracted.fcl` | stock `Production/JobConfig/cosmic/ExtractedCRY.fcl` + TimeTracker/MemoryTracker SQLite, run 1400 |
| `fcl/prof_digi_extracted.fcl` | stock `digitize/Extracted.fcl`, `Sim_best v1_5`, every event written (`SelectEvents: [DigitizePath]`) |
| `fcl/print_crvsteps.fcl` | PrintModule dump of `CrvStep` and `SimParticle` (one line each) |
| `run_profile.sh <sim\|digi> <N>` | runs one stage on a GPVM against `muse setup SimJob MDC2025ay`, outputs to `$CRVFS_DATA/profile` (`site.env`) |
| `crvfastsim g4cost` | TimeTracker SQLite -> `g4cost_stats.json` (was `analyze_timing.py`; regression-identical to `results/timing_summary.json`) |
| `crvfastsim census` | CrvStep origin census on the dumped tables (was `crv_census.py` on PrintModule text; `results/census.json` is that older output) |

Run 2026-10-15 on a Mu2e interactive machine (AMD EPYC VM, idle at start), Musing `SimJob MDC2025ay` (Offline v13_39_00, p107,
Geant4 11.3.2), 40,000 generated events, one job. Results are in `results/`; the plan quotes them.

Gotcha: the default `source.firstRun 1205` of `ExtractedCRY.fcl` hits a `Sim_best` CRV IOV with 2048 channels (pre-v04
geometry), so digitisation throws `CRVPHOTONYIELDMAKE_BAD_N_COUNTERS`. Production uses run 1400, as do these fcls.
