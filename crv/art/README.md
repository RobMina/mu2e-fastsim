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

Schema v2 validates pointer ProductIDs/keys and adds an event census. Donor dumps select compressed generators;
`dump_gen.fcl` explicitly keeps the complete `cosmicTimeOffset` stream for exposure. Old donor tables require a
new dump before building libraries; old libraries remain available for baseline comparisons. Build/validation
procedure: `../docs/generator-association-repair.md` (branch `park/tier1b-library`).
The `events` census tree also keeps the v1 branches `ngen` and `nsteps` (`nsteps=-1` with `writeSteps=false`:
not extracted, not zero) that the opt-in Cat v2 contract (branch `park/veto-closure`) reads; that contract requires a
successful full-source extraction and explicit source/extraction provenance, and legacy converter/mixer behaviour is
unchanged. The merged module (schema v2 + census v1 branches) is built and smoke-tested at `40d9339`: on 100 events
of one reference shard, `dump_gen.fcl` writes `nsteps = -1` and `ngen = n_gen` in every event and `dump_crvsteps.fcl`
writes `nsteps = n_steps`; every pre-existing branch is identical to the earlier schema-v2 and v1 dumps of the same
events, and a wrong donor generator tag still stops at the ProductID check (exit 1, no payload rows).

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
