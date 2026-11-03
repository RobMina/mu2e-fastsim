# Named-run V2 comparison

`v2compare --run-config` groups the V2 metrics by the sector types of a named run configuration. Without it the
extracted V2 state schema is unchanged. It does not establish physics closure or certify a reference campaign.

## Named V2 API

```python
from crvfastsim.analyses import v2compare

result = v2compare.run(
    ["candidate.root"], "report", compare=["reference.root"],
    run_config="run1a_v01",
    # Optional own-arm truth-crossing diagnostic:
    # steps=["candidate_steps.root"], compare_steps=["reference_steps.root"],
)
```

`run_config` accepts a registry name or a `RunConfig` object; omission retains the
legacy extracted path. This module's API is independent of main CLI wiring.
`rc.geometry()` and `rc.counters_cfg()` supply placements, bar count, layer count,
and **every** sector's `crs.sectorType*`. Pulse spectra and efficiencies aggregate
sectors of a common type; clusters use that same stock sector type. Names follow
`runconfig.sector_type_names`: Run1A/Run1B has `T1-T2` (type 1; 1,280 bars); Run2
has `R1-R6`, `L1-L3`, `T1-T5`, `E1-E2`, `U`, `D1-D4`, `C1-C2` (5,344 bars).
This is a sector-type-group comparison, not separate closure of each sector.
A `counters=` override is incorporated by replacing the named configuration's
counters reference before geometry resolution; its identity differs accordingly.

The existing tolerances and two-standard-error allowances are unchanged:
median PE 2%, spectrum band 5%, cluster rate 1%, bar-rate chi2/ndf 1.5 and maximum
absolute pull 5. Per-group results contain `status: pass|fail|unsupported`.
The overall **metric-only** outcome is `metrics_pass` / `metrics_status`. Overall named V2 is always
`pass: false, status: unsupported` until both input arms have verified configuration/time/conditions provenance:
`--run-config` currently declares the analysis layout but does not bind it to the ROOT producers. This explicit
fail-closed status prevents a metric-only agreement from being advertised as named V2 certification.
Unsupported metric evidence also fails the metric-only checks:

- Every configured group is reported, including absent/empty groups. A missing
  group is unsupported, never silently skipped.
- Median estimates require at least 50 histogram entries in **each** arm and a
  finite positive median/density estimate.
- Spectra require at least 2,000 reference entries in `[5,150)` PE, nonzero
  candidate support there, and usable equal-reference-content bins. The legacy
  rebinning and per-bin reference minimum of 100 remain unchanged; used bins are
  reported rather than asserting all 20 are populated.
- Cluster rates require nonzero cluster counts in both arms and positive event
  denominators. No clusters is unsupported rather than zero-error agreement.
- Empty bar support or nonpositive event denominators is unsupported. Per-bar
  counts still mean events with a sequence-zero pulse of at least 5 PE, counting
  each event/bar once, not each SiPM. Bars with no support in either arm do not
  contribute to chi2; `nbars_used` exposes that limitation.
- Missing events/pulses/clusters trees in **any** source file are retained in
  `missing_trees` and prevent a pass, even when other files supply those trees.
  Bad pulse bar indices, nonfinite PE, and unknown cluster types raise an error.

The layer-efficiency output is explicitly **diagnostic only**: each arm's own
primary-muon truth crossings and a fixed 8 PE pulse threshold, not extrapolated
track efficiency, not the Run2 sector-specific coincidence cuts, and not a
module-gap dead-width measurement. It does not participate in `pass`. The agreed
track-efficiency/gap-width and MC–data guard-rail rows remain unimplemented.
`metrics_pass` therefore covers only implemented metrics, not all of plan.md V2; overall `pass` remains false.

Rates default to dumped art events. Existing `generated=` and
`compare_generated=` arguments can replace those denominators; their validity
and the populations/exposure they describe are caller responsibilities. No event
pairing, duplicate-file audit, timing equivalence, conditions equivalence, or
input campaign identity is inferred from selecting a run name.

### Emit and merge

```python
v2compare.run(["candidate.root"], "unused", compare=["reference.root"],
              run_config="run2_v10", emit="batch.json")
v2compare.run(["batch.json"], "merged-report", merge=True,
              run_config="run2_v10")
```

Supply the same named configuration for merge. Named states have a
`configuration` **canonical JSON string**: an immutable snapshot of the complete
RunConfig dataclass, parsed-counters SHA256, geometry-array SHA256, sector names,
bar/layer counts, and group membership. `response_layout(rc, geom=None)` returns
a frozen `ResponseLayout` holding this identity and tuple-valued groups. No
mutable per-run module globals are used or registry objects modified.

Merges require identical identity strings before adding any counts, reject
named/legacy mixing, and validate both arms' group sets and array shapes before
accumulation. Missing-tree evidence survives merges. The summary decodes the
identity as a JSON object for readability. These are compatibility checks, not
cryptographic source authenticity or a defense against deliberately rewritten
state files. Parsed counters are hashed semantically; source comments are not.

## Local tests

```bash
PYTHONDONTWRITEBYTECODE=1 CRVFS_NO_FETCH=1 PYTHONPATH=/path/to/core:$PWD \
python3 -m pytest -q -p no:cacheprovider tests/test_named_response.py
```

The synthetic tests cover both bar counts, every sector group, unsupported support boundaries, diagnostic-only
efficiency, immutable identity/merge guards, missing trees and empty pulses. They do not run a response simulation or
claim named-run V2 closure.
