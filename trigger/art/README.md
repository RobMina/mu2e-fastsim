# R7 measured KalSeed selection exporter satellite

`R7Trigger/src/KalSeedSelectionFeatures_module.cc` is a read-only, normal art
`EDAnalyzer`, using `art::TFileService` to write ROOT TTrees for uproot. It exports
measured reconstructed inputs for the two admitted TPR KalSeed selections. It
contains **no cut emulation, filter decision, truth input or path-reach inference**.
It does not replace EventNtuple with gallery/ACLiC or require a custom dictionary:
all branches are fundamental scalars, standard strings or vectors of fundamentals.

## Scope and status

This is a new R7-owned satellite, not an Offline patch or an edit to the tracker
owner's `trkfastsim` satellite. The latter was used read-only for art/SCons idioms.
Only files below this `art/` directory belong to this implementation.

**User build, feature art execution and full measured-array postflight completed.** The
successful build is `al9-prof-e29-p109` against the exact backing below. The
library postdates all source/build inputs, and its SHA256 is
`203dce0661049cf9ee6e8fcad5d5fa1af764c667ec705c617e18c8dede617562`.
Read-only `ldd -r` on the build host reports no missing libraries or undefined
symbols; Offline resolves to `v13_41_01` and externals to the p109 view. The
registration symbols are exported. Those build checks alone do not establish actual art execution,
transferred-runtime compatibility, feature-file validity or JAX parity. The separate feature experiment
passed the worker-side pinned-library/relocation gate and completed art exit 0 with 28,044 analyzer calls.
The actual std::vector feature file passes strict structural/status checks against the saved decision-art
IDs/order. Full source/preparation/executed-configuration/baseline/selection postflight completed
successfully, with zero eager/JIT measured-array mismatches. Loaded-plugin attestation, independent
feature-extraction/boundary validation and independent-history confirmation remain open.

Build history: the first user build failed at linking
with undefined `art::detail::SharedResource_t` constructor/destructor symbols.
The exact p109 `libart_Utilities.so` was inspected with `nm -D --defined-only -C`
and exports both symbols. The missing direct dependency is now included as
`art_Utilities` in SCons and `art::Utilities` in CMake, with a source-static
regression check. The next user build reached a missing
`cet::detail::provide_file_path` reference from `DEFINE_ART_MODULE`'s
`CET_PROVIDE_FILE_PATH()` registration macro. The exact p109 `libcetlib.so`
exports that symbol; `cetlib` / `cetlib::cetlib` are now explicit direct
SCons / CMake dependencies, distinct from `cetlib_except`, with another
regression check. A read-only symbol-table audit of the user-built object found
providers for all its non-system C++ references among the updated direct
libraries, including ROOT. This is not a link or runtime test.
SCons does not infer transitive dependencies. The analyzer source is unchanged.

The separate work area's backing link points to the exact released musing.
Source/build hashes and library freshness were independently verified after the
user reported success. Those build checks alone are not reproducible-build or
execution attestation. The subsequent art/postflight result is recorded above;
strong measured-parity/runtime-closure flags remain unpromoted. There is no
setup script or binary included in this source tree.

The sole admitted backing is explicitly:

```
/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ba
```

The release-source binding is Offline `v13_41_01`, commit
`b132ab6567ac021851353e05b7a1feb45bfb7c14`. Do not substitute `HEAD`, current CI,
a current Offline checkout, or a different production release. The analyzer's
`schema` tree records a **declared** source binding, not a runtime version probe.
The caller must independently attest the actual build and loaded libraries.

## Separate user build: mu2ebuild02 only

The work area is this `trigger/art` directory, containing the single local repo
`R7Trigger`. Have the user verify that the mirror has delivered these
files and that this is a separate area, with no local Offline or owner repo.
No setup script is supplied. In a **fresh shell on mu2ebuild02**, the following
commands are a handoff for the user only, not commands executed during source preparation:

```bash
cd /path/to/r7_trig/trigger/art
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
setup muse
test "$(readlink backing)" = /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ba
muse setup
muse build -j 32
```

The local `.muse` pins `ENVSET p109`, matching the production build stub
`al9-prof-e29-p109`. Use prof, not debug or an invented current envset.
Stop if that exact backing cannot be used; do not silently migrate the build.
The user must report a completed successful build before any art validation.
`R7Trigger/src/SConscript` and `R7Trigger/CMakeLists.txt` list matching direct
Offline, KinKal, art, canvas, FHiCL, exception and ROOT dependencies; SCons does not
infer transitive libraries. No producer or data-product dictionary is added.
The CMake file follows Offline's package plugin conventions; it is not a separate
standalone CMake configure/build workflow.

## Validated FHiCL and scheduling handoff

The validated configuration uses `Atom<art::InputTag> kalSeeds` and
`Sequence<art::InputTag> selectionInfo` with these explicit defaults:

```fcl
physics.analyzers.r7SelectionFeatures: {
  module_type: KalSeedSelectionFeatures
  kalSeeds: "TrigKSFDe::TriggerReplay"
  selectionInfo: [
    "TprTrkDe80m70pD0200KSFilter::TriggerReplay",
    "TprTrkDe80m70pKSFilter::TriggerReplay"
  ]
}
```

The constructor rejects every other collection, product instance or process,
missing process qualifications, other filters, duplicate filters and sequences
not containing exactly these two filter tags. Either sequence order is allowed;
output branch slots are canonical by filter name. This prevents falling back to
stock Digitize products. It does not validate the upstream filter configuration
or prove that the two named filters consumed this collection; the separately
bound embedded replay configuration must establish that shared input.

**Do not edit the immutable completed campaign.** R7 is paused as of 2026-11-05
06:10 EST, with no R7 jobs running; see [resume gates](../PAUSE_RESUME.md). A future
authorized run must be declared a
separate feature-export campaign on the exact admitted input population/order,
with the exact production runtime and all **52 trigger paths unchanged**. Append the analyzer to the existing unconditional `DecisionOutput` endpath,
after its decision printer and before its `DecisionProducts` output. Keep all
52 trigger paths' order and evolving counters. Never insert the analyzer into a
filtering path or apply SelectEvents to its execution. Neither `trigger_paths`
nor `end_paths` changes. `feature_replay.py` prepares this separately declared
transformation only after complete hybrid baseline admission; preparation is not
build verification or launch authorization.

The new analyzer must run once for **every art source event**, including events
where reconstruction/filters never ran or produced an empty collection. Use an
unconditional endpath and the serial event traversal needed for source-order
identity. Multiple independently configured instances or multi-schedule event
reordering do not provide a unique process-wide source ordinal; use exactly one
instance and one ordered event schedule in the feature-export run.

Configure `services.TFileService.fileName` explicitly in that new run to a file
under the private data area, never the app/build area. The output TTrees
are under the analyzer label, normally `r7SelectionFeatures/events` and
`r7SelectionFeatures/schema`; read them with uproot. The analyzer does not retain
art physics products itself. Existing endpath tooling must read
`TriggerResults::TriggerReplay` separately for actual path/reach state. Do not
infer new replay reach from the stock arm, TriggerInfo absence alone, or the
presence of shared reconstructed tracks.

## Output contract (schema version 1)

`events` has exactly one row per successfully processed art event, even when the
collection and both TriggerInfo products are missing. Structural inconsistency
or a thrown released accessor aborts the job; there is deliberately no fabricated
successful row for an invalid event.

| Branches | Type / meaning |
|---|---|
| `run`, `subrun`, `event` | unsigned scalar art event identity |
| `source_ordinal` | uint64 scalar, starts at 1, increments each analyze call, does not reset at run/subrun/file boundaries; unique only within one serial process/instance |
| `product_present` | bool scalar from collection `getHandle` validity |
| `product_id` | string serialized from the actual handle's `id()` with ostream; empty only when absent |
| `track_index` | vector uint64, actual zero-based collection indices in collection order |
| `raw_status` | vector uint32, exact `status().hex()` parsed in base 16, with overflow rejection |
| `valid_track` | vector byte boolean, 1 for every actual track, no padding |
| `status_has_SeedOK` | vector byte boolean, exact `hasAllProperties(TrkFitFlag("SeedOK"))` |
| `has_t0_segment` | vector byte boolean, 1 only after status-gated `t0Segment(t0)` returns a non-end iterator |
| `supported_fit` | vector byte boolean, LoopHelix / CentralHelix / KinematicLine status support |
| `fit_particle` | vector int, measured fit hypothesis `Ks.particle()`, not truth PDG |
| `active_hits` | vector unsigned, exact `Ks.nHits(true)` for each actual track |
| `pz`, `momentum`, `tan_dip`, `d0` | vector double, exact released segment expressions below |
| `momentum_error`, `chi2_dof`, `fit_consistency` | vector double, exact released accessors/precision below |
| `t0_segment_index`, `t0` | diagnostic vector int64 / double, raw iterator offset / returned t0; unavailable index is -1 |
| `info_present_D0200`, `info_present_nominal` | separate scalar bools from `getHandle<TriggerInfo>` |
| `accepted_track_keys_D0200`, `accepted_track_keys_nominal` | vector uint64, actual TriggerInfo track keys in stored order |

All per-track vectors have the collection length. A present empty collection
has `product_present=true`, an actual ProductID and length-zero track vectors;
a missing/unrun collection has `product_present=false`, an empty ID and
length-zero vectors. The analyzer cannot distinguish missing from unrun without
separate replay path states. Byte vectors must be explicitly converted to
boolean arrays by the uproot/padding consumer, not assumed to be ROOT bool
vectors. Keep source-file/process identity with `source_ordinal` when combining
outputs; run/subrun/event alone need not be unique across files/processes.

The D0200 slot is `TprTrkDe80m70pD0200KSFilter`; nominal is
`TprTrkDe80m70pKSFilter`. An absent TriggerInfo has false presence and no accepted
keys, distinct from a present TriggerInfo accepting zero tracks. For each present
TriggerInfo, **even an empty one**, a missing shared collection throws. Every
track pointer must be nonnull, have the exact shared collection ProductID, have
an in-range key and not duplicate another key in that filter. Any violation
throws rather than changing identities or deduplicating. The same key may
legitimately appear in both filters. Presence and keys describe observed
products, not a recomputed filter boolean or minimum-track decision.

### Released expression and gate fidelity

The implementation follows `KalSeedFilter::checkKalSeed`'s status-before-t0 order:

1. Test SeedOK. If absent, do not call `t0Segment`; leave derived doubles NaN,
   `has_t0_segment=0` and diagnostic index -1. Raw identity, status, fit particle,
   support flags and hit count remain genuine measured values.
2. Call the actual `Ks.t0Segment(t0)` without catching exceptions or pre-empting
   it for empty segment vectors. In the bound release an empty segment vector
   **throws**; it is not a false gate or a fabricated empty track. An actual end
   iterator leaves the derived values unavailable. For a returned segment, save
   its actual index and returned t0.
3. Use that segment's `momentum3()` and `position3()`, then exactly
   `1.0/tan(momvec.Theta())` and `momvec.R()`. No `pz/pt` replacement, position
   propagation, nearest-z segment choice or intersection/momentum-surface
   override is admitted.
4. Use LoopHelix first, then CentralHelix, then KinematicLine, as the released
   filter does. LoopHelix d0 is
   `std::copysign(lhtraj.minAxisDist(), posvec.Cross(momvec).Z())`; the other two
   use their trajectory's `d0()`. Unsupported fits leave the measured derived
   feature doubles NaN, rather than the released local variable's initial zero.
   A returned t0/index can still be present for an unsupported fit.
5. Export `momvec.Z()`, segment `momerr()`, `Ks.fitConsistency()` and
   `Ks.chisquared()/Ks.nDOF()` with **no operand casts**. The latter divides the
   release's float chi-squared by its int NDOF **before** assignment promotes
   the result to double. Promoting operands first would change boundary values.
   Fit consistency similarly preserves the stored float value before widening.

Unavailable derived feature doubles are initialized to explicit quiet NaNs.
Downstream selection must gate on the status/segment/support fields; NaN is not
an invented zero. Supported arithmetic results, including infinities or NaNs
from a zero NDOF or pathological trajectory, are not sanitized or clamped.
The exporter does not check particle cuts, propagation direction, hit minima,
kinematic/quality thresholds, minNTrks, noFilter, upstream prescales or path
reach. Raw particle and hit-count accessors are exported for all actual tracks;
their availability does not imply a passing status gate.

The admitted two-filter contract has SeedOK, no momentum surface, no required
calorimeter association and zero stereo/plane minima. This exporter is not a
generic KalSeedFilter feature interface; extending that contract needs an
explicit source-bound schema revision and additional measured fields.

`schema` contains one row with `schema_version`, `offline_release`,
`offline_commit`, `filter_sha256`, `required_backing`, precision description and
all three encoded input tags. These are provenance declarations, not proof of
which release binary was loaded.

## Source-static checks

The source-static checks in `tests/test_exporter_source.py` passed locally. They check the
source/config/branch contract, expression and gate spelling/order, product
identity guards, absence of truth/cut emulation, and direct library-list
correspondence. They neither compile nor execute any C++ or art code:

```bash
$PROJECT_VENV/bin/python3 -B trigger/art/tests/test_exporter_source.py -v
```

## Local source binding and validation still required

The implementation used the source files from the bound public Offline release:

| File | SHA256 |
|---|---|
| `KalSeedFilter.b132ab6.cc` | `51b6dbfe6f5dcd115afdfb8b9ada7aa187b8610f717cdbbc1527e7b573fabd39` |
| `KalSeed.hh` | `fc6a22b985a7ca6b11b989152d98d5375cb15eb3912239c8aaec56a8a04176cc` |
| `KalSeed.cc` | `7c691232e4b71c280c8781aec3cc6a6dcc27ed713a38849d5fcee691cbb38f7f` |
| `KalSegment.hh` | `03d952cb8c32e02a3c8cbf44ba36c7862ecbc3d0c95189b07f203a188141d920` |
| `TriggerInfo.hh` | `cea7f516eac14986b8c06d045c0790c2305f92c2aa44877f8a8d6640300d9bff` |
| `BitMap.hh` | `fa3ff7e181fdd75674fc0e4bd44ad4265bded75d22e1fcc7a063d05a888ef595` |

The completed feature-export run passed structural/event-order, saved all-path
decision and eager/JIT measured-array checks. Retain these gates for any separately
approved future run; stronger loaded-plugin, independent-extraction, released-C++
boundary and independent-history checks remain open:

- Exact loaded-release/library/config/input-order provenance, including the
  actual shared collection consumed by both named filters.
- Rejection checks for unsupported tags and structural TriggerInfo corruption;
  coverage of missing/unrun/empty events and exceptional segment access.
- ROOT/uproot schema and vector alignment checks, finished-tree event entries
  matching the source event population (use the highest tree cycle).
- All-52 TriggerReplay path-decision equality against the saved baseline on
  identical admitted fast inputs/order; an output-only endpath is not assumed
  inert without this check. Reuse the baseline, do not rerun the before arm.
- Measured accepted-track identity/count comparison to each reached C++ filter,
  eager/JIT selection comparison, and released-C++ boundary checks, especially
  float chi2/NDOF and signed d0. Synthetic/source-static checks alone cannot
  establish these.

The user build and read-only linkage checks have passed. The separate
`feature_runtime.py` captures/rechecks source/library hashes and freezes a single
plugin plus private released-runtime runner; the worker verifies hash/stub and
relocations before art. Its immutable deployment manifest requires an independent
SHA pin, which passed in the completed experiment. The vectorized `feature_table.py`
and `feature_validate.py` check actual file structure, full event order/empty coverage, saved all-path decision parity
and eager/JIT selections. Their local fixtures alone are not actual art execution.
The separate full-history experiment completed art and full postflight on 28,044
events, with 18,038 tracks and 10,006 zero-track events. Both BO-final selections
have zero eager/JIT measured track/count/filter/composed-bit differences. This is
development measured-array evidence, not independently attested plugin loading,
independent feature extraction, released-C++ boundary certification or independent
fast confirmation; stronger source-only/readiness/parity flags remain false. The
filters cost about 0.15% of quiet trigger work; no end-to-end acceleration is shown.
See [the measured scope](../JAX_SELECTION.md).
