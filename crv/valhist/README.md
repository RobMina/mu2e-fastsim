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

Run from the umbrella with `PYTHONPATH=crv:core`: `python3 -m valhist.compare ...`, `python3 -m valhist.g4table ...`.

Outputs: `$CRVFS_DATA/valhist/<input>/<arm>/valhist.<subrun>.root` (Mu2e-internal: never committed).
