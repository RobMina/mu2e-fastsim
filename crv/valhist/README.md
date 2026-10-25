# valhist: stock-equivalent validation and DQM histograms for the CRV fastsim/fastreco

Plan and results: wiki `mu2e/wiki/sim/crv-fastsim-dqm-validation.md`. Histograms come from the stock Offline
`Validation` module (CRV classes), the CRVDQM `DqmCrvDigi`/`DqmCrvReco` clients (configuration `extracted`, binning v3)
and the legacy `DqmCrv`, run unchanged in the `crvval` muse area.

| file | role |
|---|---|
| `setup_crvval_area.sh` | make the area: Offline main CI backing build + Mu2e/DQM (PR #48 branch) + a copy of `art/CrvFastSim`; no muse command (the user builds) |
| `run_stock_g4ref.sh` | arms A0 / A0' (floor, other digitisation seed) on `g4ref_v1` shards, one GPVM, serially |
| `run_faststeps_g4ref.sh` | arm A1 (fast step tables via `CrvStepsFromTable`, gen stream) or A0g (G4 steps injected the same way: `TABLEDIR valhist/g4ref_v1/g4all_tables`, prefix `g4all`) |
| `g4table.py` | G4 CrvStep tables retyped to the fast-table column types (all steps kept) for A0g |
| `compare.py` | `merge` shard files (with `--alias` to line up product directories) and `compare` two arms: chi2/ndf, KS, totals, mean shift, pulls; floor-calibrated flags; `report.tsv`, `summary.json`, overlay PDF |
| `../art/CrvFastSim/fcl/valhist_stock.fcl` | stock CRV digitisation + reco + the three histogram sets on a G4 dts file |
| `../art/CrvFastSim/fcl/valhist_faststeps.fcl` | the same with `CrvStepsFromTable` steps, on the gen stream |
| `fasttables.py` | fast chain -> CrvRecoTableDumper-layout tables: `digis` (A2), `reco` on stock digis (A3), `chain` (A4 on G4 steps, E2E on fast steps); numpy pulse fit (chi2, LE, height, beta) and coincidence finder |
| `run_fasttables.sh` | `fasttables.py` over `g4ref_v1` shards |
| `../art/CrvFastSim/src/Crv{Digis,RecoPulses,CoincidenceClusters}FromTable` | art producers reading those tables (and stock dumps, for the round trip) into the stock collections |
| `../art/CrvFastSim/fcl/valhist_{fromtable,digisfromtable}.fcl`, `run_fromtable.sh` | table arms: all three products from the table, or digis only + stock reco |
| `run_compare_g4ref.sh` | merge each arm's shards and compare every arm with its reference (dts arms vs A0, gen-stream arms vs A0g), floor = A0' vs A0 |
| `deficit.py`, `oneside.py` | where a digi/pulse count difference between two arms sits: per sector group, SiPM charge, pulse PE and time; events with activity in one arm only, by CRY primary class |

Arms on `g4ref_v1` (outputs `$CRVFS_DATA/valhist/g4ref_v1/<arm>/`): `a0` stock reference, `a0p` floor (other digi seed),
`a0_dump` = `a0` + tables, `rt` round trip (stock tables -> *FromTable), `a0g` G4 steps injected on the gen stream,
`a1_<tables>` fast steps + stock digi/reco, `a2` fast digis + stock reco, `a3` stock digis + fast reco, `a4` fast
digis + fast reco on G4 steps, `e2e_<tables>` fast steps + fast digis + fast reco (gen stream).

Run from the umbrella with `PYTHONPATH=crv:core`: `python3 -m valhist.compare ...`, `python3 -m valhist.g4table ...`.

Outputs: `$CRVFS_DATA/valhist/<input>/<arm>/valhist.<subrun>.root` (Mu2e-internal: never committed).

## Thread B coverage and fresh pilots

The legacy `compare merge` drops flow bins and source Sumw2 on write; use original shard files for new coverage
claims. `python3 -m valhist.coverage REF CAND --outdir NEW` inventories the union, compares TH1/2/3 including flow,
separates normalization/shape descriptively, and reports missing/unsupported objects and variance gaps explicitly.
Profiles and general derived-map statistics are not silently treated as counts. `--count-key` is only for exact
paths whose fill semantics have been checked against source. See `docs/valhist-coverage.md`.

`run_pilot.sh NEW_OUT FASTC LIBRARY GEN_ART SEED` reuses retained transport in a frozen, already-built stock area;
`pilot.py` records hashes and certifies the complete event population from highest-cycle output trees.
`audit_pilot.sh` compares original pilot/reference shards. `paired_rates.py` estimates conditional event-paired
rate errors when both inputs retain all event rows; it rejects mismatched or duplicate event populations.
The executed shard-11 result and remaining gates are in `docs/thread-b-dqm-pilot.md`. No blanket DQM pass is implied.
