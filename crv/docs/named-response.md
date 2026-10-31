# Named-run response comparison and fast tables

These opt-in extensions preserve the default extracted V2 state schema and the
legacy `valhist.fasttables --calib ...` workflow. They do not establish physics
closure, certify a reference campaign, or turn a ten-event smoke test into V2.

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

## Fasttables named conditions/configuration

```bash
python3 -m valhist.fasttables chain steps.root fast.root \
  --run-config run1a_v01 \
  --conditions-manifest /data/conditions/run1a/manifest.json \
  --window offspill

python3 -m valhist.fasttables reco stock_reco.root fast_reco.root \
  --run-config run2_v10 \
  --conditions-manifest /data/conditions/run2/manifest.json
```

With `--run-config`, `--conditions-manifest` is mandatory. `--calib`,
`--deviation`, and `--counters` cannot override named assets; mismatches fail, with
no extracted or nominal fallback. The loader call is exactly
`named_conditions.load_conditions(manifest, rc, geom) -> (conditions, metadata)`.
It validates against the unmodified configured RC, **before** applying the
explicit fasttable timing policy. See [named-conditions.md](named-conditions.md)
for the strict bundle contract and evidence limitations. The named loader does
not currently export an extracted bundle; use the legacy CSV mode for extracted.

Geometry/counters come from the RC, response from `rc.chain_config(...)`, and
coincidence settings from `rc.finder()`. `--no-opposite-side` remains an explicit
control override. Existing NumPy reference reconstruction and coincidence loops
are unchanged; `--jax` accelerates response/waveforms only.

`--window offspill` is the explicit CLI default: a fixed absolute 0–100000 ns
crop, even when the named RC's configured default is onspill (200–1500 ns).
`--window onspill` requests that fixed 200–1500 ns crop. **Neither implements
stock per-event spill/evlen/pbtime semantics or establishes onspill equivalence.**
No event time rebasing, generator reweighting, or denominator correction is done.
For `reco`, no time crop is applied: the supplied stock digis already embody the
source timing policy. The output metadata explicitly records this.

Named invocations write `OUT.root.metadata.json`, containing loader evidence,
configured/effective windows, finder settings, stage/seed/backend/noise, a timing
warning, `onspill_equivalence: false`, and whether the window was applied. The
ROOT layout is unchanged. The sidecar describes settings; it is not an input or
output hash receipt or an atomic publication protocol. Optional `--write-receipt`
adds a separate prospective `OUT.root.receipt.json`; see [response-receipts.md](response-receipts.md).
Legacy invocations do not gain a sidecar and retain their extracted defaults.

Python configuration helper:

```python
from valhist.fasttables import configure
geom, counters_cfg, conditions, chain_cfg, metadata = configure(
    run_config="run2_v10", conditions_manifest="manifest.json",
    window="offspill", jax=False, noise=True, no_opposite_side=False)
```

The helper does not run simulation or reconstruction. Legacy mode takes `calib`,
optional `deviation`/`counters`, and returns `metadata=None`.

### Remaining fasttable limits

This extension does not change event labeling or population handling. The
existing step-based chain writes event rows for events present in its step table;
zero-step generated events need independent population/exposure bookkeeping. `--event-reference STOCK.root`
selects exact stock run/subrun/event identities and retains its zero-step events, for conditional A4 diagnostics;
this is not a generated-event transport acceptance comparison.
Inputs are expected to be a single run, as in the original writer. Hardware IDs
in the written digi-layout rows remain placeholder `-1`, even though response
phase grouping uses the verified conditions mapping. The NumPy pulse fitter's
existing pulse-round/fit limits remain. No claim of complete stock-product
identity follows from table compatibility or this configuration wiring.

## Optional receipt audit and diagnostics

`fasttables --write-receipt` snapshots inputs before execution and records the actual
complete chain configuration, output/event hashes and source hashes. Resolved input
paths are frozen before reads; an input symlink retarget cannot change the producer's
selected file. Existing outputs/sidecars are refused. Historical products are not
upgraded. A normal receipt cannot admit the shared-yield or physical-readout overrides.

Named V2 accepts `--candidate-receipt RECEIPT --conditions-manifest MANIFEST` for
exactly one candidate/reference pair. It revalidates the fast producer receipt and
requires the comparison reference to be byte-identical to its recorded stock
input/event-reference, with identical event identities. The expanded file paths are
frozen before metric filling; later glob matches cannot enter the comparison.
Emit/merge/generated-denominator/counters overrides are refused in this mode.
`producer_receipt_audit` is partial integrity evidence, not stock producer configuration
certification; **overall V2 remains unsupported**. Full V2 guardrails and stock binding
remain open even when the receipt and implemented metrics pass.

See named-response-diagnosis.md (branch `park/named-run2`) for the all-sector,
paired-event saved-product audit, the explicit stock shared-yield diagnostic, and
the opt-in physical readout mask. Neither changes named production defaults.

For real-backend diagnostics, `--trace-response` requires a named local step-mode
input and records strict Python call evidence before ROOT publication. Optional
`--jax --fused-response` additionally requires the trace and offspill window.
Traced/fused outputs cannot enter normal receipt admission. The default backend
and I/O policy are unchanged; see [response-execution-trace.md](response-execution-trace.md).
Fresh replay seeds can be compared with named-seed-audit.md (branch `park/named-run2`);
its spread is conditional on fixed G4/stock inputs, not full V2 uncertainty.

## Local tests

```bash
PYTHONDONTWRITEBYTECODE=1 CRVFS_NO_FETCH=1 \
PYTHONPATH=/path/to/core:$PWD \
python3 -m mu2efastsim.testselect --run --isolate -- -q -p no:cacheprovider
```

Synthetic tests cover both bar counts, every sector group, unsupported support
boundaries, diagnostic-only efficiency, immutable identity/merge guards, missing
trees and empty pulses, named configuration/conditions wiring, and legacy
compatibility. They do not execute a real response simulation, download assets,
certify conditions, or claim named-run V2 closure. Ten-event reference files are
initially smoke evidence and are expected to report unsupported metrics.
