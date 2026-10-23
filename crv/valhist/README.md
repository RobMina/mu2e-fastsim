# valhist: stock-equivalent validation and DQM histograms for the CRV fastsim/fastreco

Plan and results: wiki `mu2e/wiki/sim/crv-fastsim-dqm-validation.md`. Histograms come from the stock Offline
`Validation` module (CRV classes), the CRVDQM `DqmCrvDigi`/`DqmCrvReco` clients (configuration `extracted`, binning v3)
and the legacy `DqmCrv`, run unchanged in the `crvval` muse area.

| file | role |
|---|---|
| `setup_crvval_area.sh` | make the area: Offline main CI backing build + Mu2e/DQM (PR #48 branch) + a copy of `art/CrvFastSim`; no muse command (the user builds) |
| `run_stock_g4ref.sh` | arms A0 / A0' (floor, other digitisation seed) on `g4ref_v1` shards, one GPVM, serially |
| `../art/CrvFastSim/fcl/valhist_stock.fcl` | stock CRV digitisation + reco + the three histogram sets on a G4 dts file |

Outputs: `$CRVFS_DATA/valhist/<input>/<arm>/valhist.<subrun>.root` (Mu2e-internal: never committed).
