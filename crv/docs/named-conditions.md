# Strict named-conditions assets

`crvfastsim.named_conditions` exports the **actual stock-job conditions evidence**
for `run1a_v01` or `run2_v10`; Run1B remains outside this admission contract. It does not resolve a database IOV,
substitute extracted conditions, generate nominal constants, or invent an FEB map.
The supported source is an **explicit text-file all-run overlay on
`Sim_best/v1_5`**, observed in one stock job at reference run **1400**. The database
contents at that run are not claimed to be the named geometry's conditions.

## Producer contract

Prepare one local source directory per configuration, containing these exact files:

| File | Required evidence |
| --- | --- |
| `crvreco.root` | `crvdump/calib`: first-event `run,channel,pedestal,pulsearea,pulseheight,timeoffset,status,roc,feb,febch`; `crvdump/events`: `run,subrun,event,spill,evlen,pbtime` |
| `resolved.fcl` | Fully resolved stock-job configuration, not an unresolved include/override deck |
| `manifest.txt` | Original backing-file SHA256 receipts and any additional job provenance |
| `calib_source.txt` | Copied original `TABLE CRVTime` and `TABLE CRVSiPM` |
| `status_source.txt` | Copied original sparse `TABLE CRVBadChan` |
| `photonyield_source.txt` | Copied original dense `TABLE CRVPhoton` |
| `ordinal_source.txt` | Copied original whitespace `Channel ROC FEB FEBchannel` header and mapping |
| `stock.log` | Retained stock execution log |

The backing snapshots must be copied from the actual files used by the job, not a
similarly named local mirror. `manifest.txt` must contain exactly one matching
`sha256sum` receipt per configured source: `64_HEX_DIGITS  /absolute/backing/path`.
The path must equal the configured reference or end in `/` followed by its entire
relative path, such as `/Offline/CRVConditions/data/photonyield_run1a_v01.txt`.
Comments or additional provenance lines are allowed. Conflicting or repeated
matching receipts fail. Receipts bind bytes and configured references; they are
not cryptographic proof of CVMFS origin or execution success.

`resolved.fcl` must contain:

- `services.DbService.purpose: "Sim_best"` and `version: "v1_5"`.
- Exactly three `services.DbService.textFile` entries, with basenames
  `calib_<geometry>.txt`, `status_<geometry>.txt`, and
  `photonyield_<geometry>.txt`.
- `services.ProditionsService.crvCalib.useDb`, `crvStatus.useDb`, and
  `crvPhotonYield.useDb` all `true`.
- `services.ProditionsService.crvOrdinal.filePath`; the stock ordinal maker
  appends `/<geometry>.txt` to this directory.

Here `<geometry>` is `run1a_v01` for Run1A and `nominal_v10` for Run2.
Both nested resolved tables and dotted parameter paths are supported. Duplicate
paths, unresolved references and unsupported syntax fail, rather than attempting
general FHiCL evaluation. Other overlays, perfect-yield substitutions, and renamed
source variants are deliberately unsupported by this version.

## Export and load

Resolve the named counters file through the existing `RunConfig` machinery, then:

```bash
python3 -m crvfastsim.named_conditions export \
  --source /data/stock/run1a --run-config run1a_v01 \
  --outdir /data/conditions/NEW_run1a
```

The output parent must exist. An existing output, including an empty directory or
a dangling symlink, fails. Validation runs before output creation; a subsequent
I/O failure can leave a partial bundle, which must not be treated as complete.
No overwrite, cleanup, resume, network database query, or stock job is performed.
Use new data directories; do not commit calibration payloads to git.

The bundle contains `calib.csv`, `photon.csv`, `counters.txt`, `manifest.json`, and
an `inputs/` directory with byte-for-byte copies of **all eight** producer files,
including ROOT and logs. This deliberately makes the evidence portable but can
duplicate a large ROOT file. CSVs have commented headers and are compatible with
`chain.Conditions.from_csv(calib, photon)`; strict consumers should use the loader:

```python
from crvfastsim import named_conditions, runconfig

rc = runconfig.get("run1a_v01")
geom = rc.geometry()
conditions, metadata = named_conditions.load_conditions(
    "/data/conditions/NEW_run1a/manifest.json", rc, geom)
```

`export_conditions(source, rc, outdir, geom=None)` is the corresponding Python
export API and returns the manifest `Path`. `metadata` is the parsed, verified
manifest. Loader configuration matching is exact, including window and response
settings; passing a modified `RunConfig` fails, even if its name is unchanged.
The counters reference is part of that identity; relocated backing files are
allowed when the configured reference and bytes remain unchanged.

## Validation and provenance

- Require nonempty unique stock events from exactly one run/subrun, run 1400,
  and the same run in the first-event calibration dump. Record first/min/max
  event, count, full ordered event-identity digest, and all six event-column digest.
  Event count is **dumped stock events**, not generated exposure. Conditions are
  read once per job; this is not per-event or multi-IOV conditions evidence.
- Require exactly `0..4*geom.nbars-1`, once each, for stock calibration, both dense
  calibration source tables, and photon deviations. Sorting is allowed; dropping,
  repeating, truncating, or silently filling dense rows is not.
- Require finite calibration/deviation values, positive pulse-area/height gains,
  and integral nonnegative status. Sparse status omissions mean zero. No extra
  source table or directive is silently ignored; `TABLE` declarations and numeric
  row widths are schema-specific. Comments start with `#`.
- Compare calibration and photon values at stock float32 precision. Check
  channel/status/hardware identities exactly, not with numerical tolerance.
  Export calibration values in their float32 representation, promoted to float64
  for CSV round trips. Overflow/underflow that invalidates gains fails.
- Check ROC 1–18, FEB 1–24, FEBchannel 0–63 and reject duplicate mapped addresses.
  Only `(-1,-1,-1)` denotes missing hardware, and only where the source ordinal
  omitted that channel and the stock dump also recorded that sentinel. Partial
  sentinels and invented block-based FEB assignments fail. The CSV retains
  FEBchannel even though `Conditions` uses only ROC/FEB for phase grouping.
- Bind all input and output file hashes/sizes, resolved source references and
  backing paths, full dataclass run-config hash, counters bytes, and ordered
  geometry arrays/sector names. Reconstruct geometry from counters to reject a
  mismatched `geom` argument. Recheck inputs against concurrent drift.
- The loader rehashes all evidence, validates config/counters/geometry, reparses
  source files and resolved bindings, rereads stock ROOT, and closes both CSVs
  against sources. It does not fall back to `nominal_conditions` or default photon
  deviations. It creates `Conditions` only after all checks pass.

**Photon limitation:** the stock calibration tree does not dump photon yield.
Photon evidence is therefore the exact dense table bound to the actually
configured text overlay with `crvPhotonYield.useDb=true`, not an independent
stock numerical photon measurement. The manifest labels this explicitly.

A manifest is an integrity receipt, not an externally signed certificate: an actor
able to replace all evidence and hashes can create a different consistent bundle.
The exporter does not certify the log's exit status, DB contents, production
approval, physics closure, or FHiCL runtime plugin loading. Preserve independent
campaign evidence for those claims.

## Tests

Use the existing Python environment with CRV and compatible core on `PYTHONPATH`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/path/to/core:$PWD CRVFS_NO_FETCH=1 \
$PYTHON -m pytest -p no:cacheprovider -q \
  tests/test_named_conditions.py
```

Tests use only synthetic temporary payloads. They cover CSV round trips, float32
semantics, sparse status/missing hardware, full-width event identities, exact
coverage and duplicate rejection, invalid values/headers/tables, wrong photon
configuration, backing receipt mismatch, hash and identity drift, manifest paths,
exclusive publication, and the module CLI. They do not certify a real stock export.
