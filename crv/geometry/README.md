# V0 geometry check

The fast simulation builds the CRV bars from Offline's own counters file (`crvfastsim.geometry.from_offline_config`,
a re-implementation of `GeometryService/src/CosmicRayShieldMaker.cc`), so a module move is a parameter edit. V0
checks it bar by bar against the Geant4 world that Offline builds.

| file | what it does |
|---|---|
| `fcl/dump_geometry.fcl` + `fcl/geom_dump_extracted_v04.txt` | stock `ExtractedCRY.fcl` world, version pinned to v04, 1 event; writes the whole world to GDML and prints every bar placement (`crs.verbosityLevel 5`) |
| `crvfastsim v0check` | compares the re-implementation with the GDML (double precision) and the printout (bar-index order); writes `v0_check.json` and, with `--table`, the bar table CSV |

Run (GPVM; the geometry text file is found through `MU2E_SEARCH_PATH`, not `FHICL_FILE_PATH`):

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh; muse setup SimJob MDC2025ay
source site.env; G=$CRVFS_CODE/geometry; O=$CRVFS_DATA/geometry
cd $O && export FHICL_FILE_PATH=$G/fcl:$FHICL_FILE_PATH MU2E_SEARCH_PATH=$G/fcl:$MU2E_SEARCH_PATH
mu2e -c dump_geometry.fcl -n 1 > dump.log 2>&1
pyenv ana; cd $CRVFS_CODE
python3 -m crvfastsim v0check $O/mu2e_extracted_v04.gdml --g4log $O/dump.log --outdir $O --table $O/crv_bars_extracted_v04.csv
```

## Result (2026-10-16, Musing SimJob MDC2025ay = Offline v13_39_00): **pass**

| check | result |
|---|---|
| bars: re-implementation / GDML / printout | 576 / 576 / 576 |
| (sector, module, layer, counter) at every bar index, against the G4 construction order | identical |
| bar centres, max \|residual\| | 3.6e-12 mm (threshold 1e-3 mm) |
| bar half lengths, max \|difference\| | 0 |
| thickness/width/length axes | identical for all bars |
| rotated placements on any path to a bar | none |
| Mu2e origin in the G4 world frame | (0, 0, 0) for this world |
| printout cross-check (1-decimal print) | max 0.05 mm, the print rounding |

Outputs (data area): `$CRVFS_DATA/geometry/{mu2e_extracted_v04.gdml (3.7 MB),
dump.log, v0_check.json, crv_bars_extracted_v04.csv}`; copies of the JSON and CSV are in `results/`.

The GDML also holds the passive material (building, hatch blocks, G-blocks, CRV aluminium sheets and strongbacks)
that the muon transport needs next.
