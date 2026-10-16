# art helpers (muse satellite area)

`CrvFastSim/` is a muse package (a `.muse` file plus `src/SConscript`) built in its own work area, backed by the CVMFS
Musing, so `off/` is not touched:

| module | role |
|---|---|
| `CrvStepTableDumper` | CrvSteps, SimParticles, GenParticles (+ CosmicLivetime per subrun) -> flat TTrees `steps`, `particles`, `gen`, `subruns`. uproot cannot read these art products directly (`CrvStep` strided-size mismatch, memberwise `SimParticle`/`GenParticle`) |

Work area: this `art/` directory itself (`setup_area.sh` adds the `backing` link to the CVMFS Musing `SimJob MDC2025ay`;
`build/` and `backing` are git- and sync-ignored). `CrvFastSim` must stay a real directory: muse skips symlinked repos.
Build (on a Mu2e build machine):

```bash
cd $CRVFS_CODE/art && source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && setup muse && muse setup && muse build -j 32
```

Run: `mu2e -c CrvFastSim/fcl/dump_crvsteps.fcl -s dts....art` (and `dump_gen.fcl` on the gen stream).

## CrvRecoTableDumper (fast CRV reco reference)

| module | role |
|---|---|
| `CrvRecoTableDumper` | `CrvSiPMCharges`, `CrvDigiMC` voltages, `CrvDigi` ADCs, `CrvRecoPulse`, `CrvCoincidenceCluster` and the per-channel `CRVCalib`/`CRVStatus`/`CRVOrdinal` -> flat TTrees in `crvdump/` (uproot cannot read these products: memberwise streaming) |

Jobs (no art output; `dump_crvreco_ref.sh mc|data`, outputs `$CRVFS_DATA/reco_ref/`):
- `CrvFastSim/fcl/dump_crvreco_from_dts.fcl`: stock CRV-only extracted digitisation + CRV reco on a dts file (MC).
- `CrvFastSim/fcl/dump_crvreco_from_dig.fcl`: CRV reco on KPP data digis (unpack first with the KPP unpack fcl on
  `CRVFS_FCL_PATH`).
Readers and comparisons: `crvfastsim/reco/` (`python3 -m crvfastsim.reco ...`).
