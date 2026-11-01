# Input release and verification

The fast chains read Mu2e-internal inputs that are not in these repositories. This page says what a release must
contain and how to verify one; [`docs/running.md`](../docs/running.md) says how to run once you have it.

## Availability

A fresh clone is not a data release. **No collaboration-readable input release is published yet**, and where it
will be published is still to be decided. Until then, obtain an approved, versioned manifest and source from the input
publisher. Do not link another user's private data area, assume its permissions, or use a scratch cache as the
canonical release. The Apache-2.0 code licence does not grant rights to conditions, simulation-derived banks, detector
data, CRY data, or an image's third-party dependencies.

Proposed release layout (paths relative to `$M2FS_INPUTS`):

| Path | Needed by | Reproduction/provenance |
|---|---|---|
| `crv_fastsim/tier1b_strict_v2/{library.root,topology_sidecar_strict_v2_v2.npz,topology_sidecar_strict_v2_v2.npz.json,templates_v3.npz,templates_v3.npz.json}` | default CRV tier1b-apply, runchain, e2e-bench, kit | crv `docs/tier1b-default.md` (pinned SHA-256 per file); donor G4 production and versioned builders |
| `crv_fastsim/geometry/mu2e_extracted_v04.gdml` | CRV and tracker transport | CRV geometry/ Offline export; pin release and configuration |
| `crv_fastsim/geometry/{crvcalib_sim_best_v1_5_run1400.csv,crvphoton_sim_best_v1_5_run1400.csv}` | CRV response/reco | simulation run-1400 conditions; CRV docs/response-configuration.md; export needs conditions access |
| `crv_fastsim/lookup_v6_0/{LookupTable_6000_0,LookupTable_2370_1,LookupTable_3200_0,photonMap.root}` | CRV response (not tier1b-apply alone) | CVMFS DataFiles/CRVConditions/v6_0; names from extracted response configuration; Mu2e-internal, publisher approval needed |
| `offline/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt` and `offline/Offline/CRVResponse/data/singlePEWaveform_v3.txt` | CRV geometry and waveform | public Mu2e/Offline tag v13_39_00; container/fetch_offline_data.sh; included in standard image |
| `crv_fastsim/g4ref_v2_tables/gen.001400_00000022.root` | one-shard CRV/tracker smoke | extracted CRY reference shard 22, 50000 generated events; crv/grid and art/dump_g4ref.sh; use only a manifest-selected shard |
| `trk_fastsim/lib/trklib_mdc2025as_20f_v2.root` (+ metadata) | optional tracker secondaries | tracker buildlib format 2; G4 production donors; not read by primary-only steps |
| `trkfs_digi_20261026/v2/digicond.npz` | tracker_point digi | tracker digi-cond from FHiCL/conditions/GDML; geometry/release must match |
| `trk_fastreco/cond_mc_au.json` and `trk_fastreco/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root` (conditions tree only) | tracker_point reco/drift | tracker reco-conditions from matched dumper; kit_task.INPUT_PATHS is authoritative; no stock event conditioning |
| `beam_fastsim/g4ref_full/g4ref.npz` | beam point | MuBeamCat input and beam/art/run_g4ref.sh, beamfastsim g4ref and reference audit; no public redistributable source established |
| beam `offline/` geometry include tree, `maps/DSMap.{header,bin}` | beam point | beamfastsim.stage: Offline v10_47_00 and DataFiles/BFieldMaps/Mau13; pin every fetched file |

Native CRY is **opt-in**, not the default. It removes the gen tables only, not the secondary bank, geometry, optics
or conditions; it needs the CRY-enabled runtime and licensed data (`core/docs/native-cry.md`). Geant4 comparisons
still need matching reference tables. The named (non-extracted) configurations need a tier-1b library that does not
exist today; do not substitute the extracted bank.

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

Alternatively use `--base-url https://publisher.example/release` for approved
HTTPS objects. This tool does not mint credentials or implement Rucio/dCache
authentication: use the experiment's approved client to populate a source release,
then verify it. Fetch refuses mismatched existing files, traversal and symlink
escapes; verifies temporary bytes and publishes without overwriting. Do not use
it in a directory concurrently writable by untrusted users. Exit codes: 0 all
verified/fetched, 1 missing/mismatched or dry-run work remains, 2 invalid manifest,
I/O error or refused fetch. Missing hashes are never treated as verification.
