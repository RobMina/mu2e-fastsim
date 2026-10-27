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

The new, **unbuilt** `CrvStepTableDumper` census addition writes `events` with one
`(run, subrun, event, ngen, nsteps)` row per analyzed event, including empty generator
and step collections. With `writeSteps=false`, it still writes the census and uses
`nsteps=-1` (unknown/not extracted, not zero). The opt-in [Cat v2 contract](../docs/veto-cat-v2.md)
requires a successful full-source extraction and explicit source/extraction provenance;
legacy converter/mixer behavior is unchanged. This edit has only synthetic Python
tests, no compiled-art validation. The user must run the full build above on
**mu2ebuild02** and confirm it before an art smoke test; no new build dependencies.

## CrvRecoTableDumper (fast CRV reco reference)

| module | role |
|---|---|
| `CrvRecoTableDumper` | `CrvSiPMCharges`, `CrvDigiMC` voltages, `CrvDigi` ADCs, `CrvRecoPulse`, `CrvCoincidenceCluster` and the per-channel `CRVCalib`/`CRVStatus`/`CRVOrdinal` -> flat TTrees in `crvdump/` (uproot cannot read these products: memberwise streaming) |

Jobs (no art output; `dump_crvreco_ref.sh mc|data`, outputs `$CRVFS_DATA/reco_ref/`):
- `CrvFastSim/fcl/dump_crvreco_from_dts.fcl`: stock CRV-only extracted digitisation + CRV reco on a dts file (MC).
- `CrvFastSim/fcl/dump_crvreco_from_dig.fcl`: CRV reco on KPP data digis (unpack first with the KPP unpack fcl on
  `CRVFS_FCL_PATH`).
Readers and comparisons: `crvfastsim/reco/` (`python3 -m crvfastsim reco-<cmd> ...`).

## *FromTable producers (stock-equivalent validation, `valhist/`)

| module | role |
|---|---|
| `CrvDigisFromTable` | `crvdump/digis` rows of the event -> `CrvDigiCollection` (ROC/FEB from the table, or from `CRVOrdinal` when -1) |
| `CrvRecoPulsesFromTable` | `crvdump/pulses` -> `CrvRecoPulseCollection` in table order (waveform indices into the digis of the same table) |
| `CrvCoincidenceClustersFromTable` | `crvdump/clusters` -> `CrvCoincidenceClusterCollection`; member pulse indices -> `art::Ptr` into `pulseTag` |

Tables: `valhist/fasttables.py` (fast chain) or a `CrvRecoTableDumper` file (round trip). Jagged columns may be
`std::vector` branches or uproot counted arrays (`CrvRecoTable.hh`).
