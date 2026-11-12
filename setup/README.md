# Input release and verification

The fast chains read Mu2e-internal inputs that are not in these repositories. This page says what the input release
contains and how to verify it; [`docs/running.md`](../docs/running.md) says how to run once you have it.

## Availability

A fresh clone is not a data release. The current input release is **`mu2e-fastsim-inputs-v1.2`** (78 files, about
2.4 GB; it supersedes v1 and v1.1, which stay unchanged). It is on Fermilab shared storage, readable by the Unix group `mu2e`
from a Mu2e GPVM and from an EAF server, which mounts the same storage. Nothing is published on CVMFS or dCache, and
the release must not be copied outside Mu2e: every file is marked `redistribution: mu2e-internal`. The Apache-2.0
code licence does not grant rights to conditions, simulation-derived banks, detector data, CRY data, or an image's
third-party dependencies.

**Requesting access.** You need a Fermilab computing account in the Mu2e group (the Unix group `mu2e` on the Mu2e
GPVMs; EAF uses the same accounts). Ask the fast-simulation maintainers, through a GitHub issue on this repository or
the Mu2e simulation channels, for the release root and the manifest path of `mu2e-fastsim-inputs-v1.2`. Neither is
written in this public repository, because the manifest's provenance names internal storage. Then point
`M2FS_INPUTS` at the release root and verify it against its manifest (below) before the first run.

The format of the manifest is shown, with placeholder values, in [`RELEASE.example.json`](RELEASE.example.json).

## Release v1.2 layout

Paths are relative to `$M2FS_INPUTS`. `site.env.example` derives `CRVFS_DATA`, `CRVFS_LOOKUP_DIR`,
`FASTSIMKIT_*_DATA` and `MU2E_SEARCH_PATH` from it.

| Path | Read by | Provenance |
|---|---|---|
| `crv_fastsim/tier1b_strict_v5/{library.root,topology.npz,topology.npz.json,templates_v3.npz,templates_v3.npz.json}` | default CRV chain (`runchain`, `tier1b-apply`, `e2e-bench`), kit `chain_point`, EAF `chain` and `regression` | crv `docs/tier1b-default.md`; library SHA-256 pinned in `crvfastsim.tier1b_assets` |
| `crv_fastsim/tier1b_run1a_v01/{library.root,topology.npz,topology.npz.json,templates_v3.npz,templates_v3.npz.json}` | opt-in Run 1A CRV chain (`--run-config run1a_v01 --tier1b run1a_v01`) | the same builder on the Run 1A geometry, per primary muon; library SHA-256 pinned in `crvfastsim.tier1b_assets` (new in v1.1) |
| `crv_fastsim/geometry/mu2e_extracted_v04.gdml` | CRV and tracker transport (default `extracted_v04`) | GDML world export, crv `geometry/README.md` |
| `crv_fastsim/geometry/{mu2e_run1a_v01.gdml,mu2e_run2_v10.gdml}` | named run configurations (`crvfastsim.runconfig`) | GDML world exports |
| `crv_fastsim/geometry/{crvcalib,crvphoton}_sim_best_v1_5_run1400.csv` | CRV response and reco | simulation conditions `sim_best_v1_5`, run 1400 |
| `crv_fastsim/conditions_run1a_v01/{crvcalib_run1a_v01.csv,crvphoton_run1a_v01.csv}` | Run 1A CRV chain (`--calib`, `--deviation` with `--run-config run1a_v01`) | Offline Run 1A CRV conditions converted by crv `scripts/offline_conditions_csv.py`; required: `runchain` refuses `--calib nominal` for `run1a_v01`/`run1b_v01`, which would count the 45 ignore-in-reconstruction channels as live (new in v1.2) |
| `crv_fastsim/lookup_v6_0/{LookupTable_6000_0,LookupTable_2370_1,LookupTable_3200_0,photonMap.root}` | CRV response (`CRVFS_LOOKUP_DIR`), kit `response_bench`, `v4_response` | CVMFS `DataFiles/CRVConditions/v6_0` |
| `crv_fastsim/g4ref_v1_tables/gen.001400_000000{01..20}.root` | kit `chain_point` (`gen_set` 0), EAF `chain` (held-out 11-20), `tracker_point` (default shard 11) | CRY primaries in the extracted world, 50,000 generated events each |
| `crv_fastsim/g4ref_v1_tables/crvsteps.001400_00000012.root` | EAF `regression` (V4 SiPM on Geant4 steps) | Geant4 CrvSteps of subrun 12 |
| `crv_fastsim/v2_fastC_v1/ref.001400_000000{11..20}.root` | optional `--compare` of EAF `chain` | stock chain on the same primaries (V2) |
| `crv_fastsim/g4ref_v2_tables/gen.001400_00000022.root` | one-shard CRV/tracker smoke | extracted CRY reference shard 22 |
| `crv_fastsim/prod_as_tables/crvsteps.001400_{00000990,00001023,00001056,00001089,00001122}.root` | kit `crv_response` (`tables=vr5`) | Geant4 production CrvSteps, MDC2025as |
| `crv_fastsim/background/fit_v1/bg.npz`, `crv_fastsim/timecal/corr_{P4,P3}_v2mod.json`, `crv_fastsim/r3/dataref/dataref_{P4,P3,3runs}.json` | kit `crv_response` (background model, period time calibration, data reference) | fits to KPP data |
| `crv_fastsim/veto_ve1/run1a_dev_sel.parquet`, `crv_fastsim/deadtime_run1a/ce_mix1bb_dev_sel.parquet` | kit `veto_c4` | cached Run 1A selections (`crvfastsim.veto`) |
| `trkfs_digi_20261026/v2/digicond.npz` | `tracker_point` digitisation | tracker digi conditions from FHiCL, conditions and GDML |
| `trk_fastreco/cond_mc_au.json`, `trk_fastreco/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root` (conditions tree only) | `tracker_point` hit reco and drift calibration | stock `StrawHitTableDumper` (tracker `art/`, `dump_tracks_mc_saveall.fcl`, conditions `Sim_best` v1_5, extracted v04 geometry) on the MDC2025au CRY-extracted dig file of subrun 1400_0, then `python3 -m trkfastsim reco-conditions` (new in v1.1) |
| `trk_fastsim/lib/trklib_mdc2025as_20f_v2.{root,json,meta.json}` | optional tracker secondaries | buildlib format 2, 20 Geant4 donor files |
| `beam_fastsim/g4ref_full/g4ref.npz` | kit `beam_point` | DS-entrance muons from MuBeamCat MDC2025ac |
| `beam_fastsim/g4ref_run1baa/g4ref_run1baa_vd8.npz` | kits `beam_sob`, `beam_edep` | Coll5_Out-plane beam sample from MuBeamCat Run1Baa |
| `beam_fastsim/edepana/shower_library_cal2.npz`, `beam_fastsim/edepana/edge_library_mdc2025ac.npz`, `beam_fastsim/edepana/mu2e_run1a_v01.gdml` | kit `beam_edep` | calorimeter whole-event shower library (Geant4 CeEndpoint disk-front entries, `calofastsim.showers`); Geant4 inner-edge calorimeter energies (MDC2025ac CeEndpoint, first VD crossing on the r = 335 mm edge cylinder, with the EdepAna event energies); the Run 1A GDML for the tracker transport |
| `beam_fastsim/centuple/` (`shower_library.npz`, `edge_library.npz`, `tracker.gdml`, `calo_response/`, `digicond_onspill.npz`, `trigger/{replay.fcl,trgdump.root,TimeCluster.weights.xml,response.json}`, `donor/{donor.npz,nts.*.root}`) | kit `beam_centuple` | the edepana calorimeter inputs, the calorimeter response inputs, on-spill tracker digitiser conditions, the prepared trigger menu with its hit conditions, time-cluster MLP and response layer, and the donor library with its stock development EventNtuple files |
| `offline/Offline/Mu2eG4/geom/crv_counters_{extracted_v04,run1a_v01,v10}.txt`, `offline/Offline/CRVResponse/data/singlePEWaveform_v3.txt` | CRV geometry and waveform (`MU2E_SEARCH_PATH`) | public Mu2e/Offline v13_39_00 |

**Not in v1.2:**
- There is no tier-1b library for `run1b_v01` or `run2_v10`; those configurations run only with `--library`.
- The beam field maps and the Offline geometry include tree are public. `beamfastsim` reads them from CVMFS, or
  `python3 -m beamfastsim.stage` fetches them where there is no CVMFS mount (`docs/running.md`).
- The optional gen sets `crygen_*_tables` (`chain_point` `gen_set` 1-3) are not included.

Native CRY is **opt-in**, not the default. It removes the gen tables only, not the secondary bank, geometry, optics
or conditions; it needs the CRY-enabled runtime and licensed data (`core/docs/native-cry.md`). Geant4 comparisons
still need matching reference tables.

## Verify before use; fetch is explicit and dry-run by default

The manifest is JSON schema version 1. Every file records `path` (safe relative
path), `size_bytes`, `sha256`, `provenance`, `redistribution` (`approved` only
after review). It must be supplied through a trusted publisher; a checksum does
not authenticate a malicious manifest. `inventory` hashes explicit sources and
records metadata, without copying inputs. Do not commit internal manifests or
source paths to the public repository.

```bash
python3 -m mu2efastsim.inputs check-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS"
python3 -m mu2efastsim.inputs fetch-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS" \
  --source-dir "$APPROVED_SOURCE"                     # dry-run; exit 1 if anything is missing
# Only after source, location and redistribution approval:
python3 -m mu2efastsim.inputs fetch-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS" \
  --source-dir "$APPROVED_SOURCE" --execute
```

`check-inputs` reads every file once; for v1.2 that takes about 10 s on a GPVM or an EAF server. Read the release
in place on FNAL storage: its files are marked `mu2e-internal`, so `fetch-inputs --execute` refuses to copy them.

Access check for a new account, with `MANIFEST` and `M2FS_INPUTS` set. It reads the manifest, then one file of each
kind, then verifies everything; expect `mu2e-fastsim-inputs-v1.2 78 files`, seven `ok` lines and `78`:

```bash
python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["release"], len(m["files"]), "files")' "$MANIFEST"
for f in crv_fastsim/tier1b_strict_v5/library.root crv_fastsim/geometry/mu2e_extracted_v04.gdml crv_fastsim/lookup_v6_0/photonMap.root crv_fastsim/g4ref_v1_tables/gen.001400_00000011.root trk_fastreco/cond_mc_au.json beam_fastsim/g4ref_full/g4ref.npz offline/Offline/CRVResponse/data/singlePEWaveform_v3.txt; do head -c 1 "$M2FS_INPUTS/$f" > /dev/null && echo "ok $f" || echo "DENIED $f"; done
python3 -m mu2efastsim.inputs check-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS" | grep -c '"status": "ok"'
```

Alternatively use `--base-url https://publisher.example/release` for approved
HTTPS objects. This tool does not mint credentials or implement Rucio/dCache
authentication: use the experiment's approved client to populate a source release,
then verify it. Fetch refuses mismatched existing files, traversal and symlink
escapes; verifies temporary bytes and publishes without overwriting. Do not use
it in a directory concurrently writable by untrusted users. Exit codes: 0 all
verified/fetched, 1 missing/mismatched or dry-run work remains, 2 invalid manifest,
I/O error or refused fetch. Missing hashes are never treated as verification.
