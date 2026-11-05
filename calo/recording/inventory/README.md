# Optional native placement inventory — compact v2 review

**Compact v2 is user-built and structurally audited on a successful zero-transport initialization.** Its complete
all-world placement inventory has 167,623 placements, 175,997 rows and 49,035,617 bytes within the unchanged
256 MiB fixture cap. All 13 candidate roots occur once. There are 1,143 unresolved-solid placement occurrences,
including the G4UnionSolid CalorimeterMother; no containment/navigation or material-composition certificate.
The earlier v1 attempt failed at placement 163,069 under the cumulative file cap and remains historical evidence.
Only representation/helper allocation changed, not geometry/physics/caps. No transport is authorized here.

## Artifacts and reproduction

- `payload/Mu2eG4PlacementInventory.hh`: header-only exporter, same optional BeginOfRunAction hook.
- `placement-inventory.patch`: new inventory header plus existing Mu2eG4RunAction include/hook only.
- `source-manifest.json`: exact audited baseline/reference, patch and patched hashes; not a runtime certificate.
- `make_patch.py`: regenerates artifacts here from an explicitly selected, read-only baseline checkout.
- `check_patch.py`, `tests/`: source/guard assertions, source-drift rejection, temporary exact apply/reverse and
  boundary-adapter coexistence. No compile, native geometry or runtime memory test.
- `calofastsim.placement_inventory`: independent bounded v2 structural reader; not a native geometry validator.
- `VALIDATION.txt`: source-test receipt; older native v1 results are recorded separately.

```sh
PYTHONDONTWRITEBYTECODE=1 "$PROJECT_VENV/bin/python3" \
  calo/recording/inventory/check_patch.py --offline "$OFFLINE_CHECKOUT"
```

The selected checkout must match the manifest. Do not point this baseline patch checker at the assembled area
where the header/hook are already installed. All actual patch applications in these tests use disposable copies.
Regeneration after reviewed baseline changes uses `make_patch.py --offline "$OFFLINE_CHECKOUT"`; drift is never
an automatic waiver. The isolated assembled source and its exact-CI overlay have separate source/build receipts.

## Unchanged integration and topology scope

The same four optional `Mu2eG4Config::Debug` settings are required; no config or dependency change:

| Setting | Default | Meaning |
|---|---:|---|
| `boundaryRecorderPrefix` | `""` | Empty: no traversal/output. Nonempty: boundary diagnostic and inventory enabled. |
| `boundaryRecorderRoots` | `[]` | Explicit unique exact native PV names; unique in store and expanded world. |
| `boundaryRecorderMaxRows` | `1000000` | Whole-file JSONL rows, **including dictionary records and both markers**. |
| `boundaryRecorderMaxBytes` | `1073741824` | Whole-file bytes including newlines/footer, hard ceiling 1 GiB. |

Hook remains after `_steppingAction->finishConstruction()` in BeginOfRunAction, before navigator diagnostics.
Sequential G4 only; MT applications are rejected even with one worker. The sole navigator observation is
`GetNavigatorForTracking()->GetWorldVolume()`. No point location/relocation, RNG, process/track mutation,
parameterised-copy evaluation or geometry change is added.

Every expanded world occurrence is retained, including all noncandidate branches. Ordinary shared daughters
beneath differently placed ancestors remain separate placement records. No unique-PV/LV dictionary hit suppresses
traversal. Active-pointer cycles, null nodes/LV/material/solid, replicas, parameterisations and non-placement
physical types anywhere fail closed. Candidate names require exactly one physical-store match **and** exactly
one expanded occurrence; a store-only root or shared root under two placed ancestors fails.

Full-path uniqueness is checked without retaining all full paths: a single world and unique native `(name,copy)`
children for every parent imply unique complete paths by induction. The sibling set compares full native name
bytes, not delimiter-joined or truncated strings; it is released before recursion. A branch-local active-pointer
set detects cycles without rejecting legal reuse after branch exit. Name encoding remains JSON `U+00xx` escaping
of non-ASCII native bytes, not UTF-8 decoding.

## v2 record and identity contract

Schema is explicitly **`mu2e-placement-inventory-v2`**, incompatible with v1 inline paths/descriptions. No v1
consumer may silently reinterpret it. IDs are zero-based, inventory-local, deterministic first encounters in
native daughter preorder; they are not stable identifiers across files/geometry versions. Native pointers are
used only for in-memory interning and never serialized or sorted to assign IDs. Distinct pointers with equal
names/parameters remain distinct dictionary entries.

Record order:

1. One `inventory_begin`, including limits, root names, scope, units and frame.
2. All `material` records: `id`, native name and index.
3. All `solid` records: `id`, native name/entity, bounded supported parameters/text or unsupported metadata.
4. All `logical_volume` records: `id`, name, `material_id`, `solid_id`.
5. All expanded `placement` records in preorder: `id`, `parent_id` (`null` only for world), depth, PV name/copy,
   `logical_volume_id`, local translation and row-major 3x3 object rotation, `selected`, `candidate_root_id`.
6. One `inventory_end` with dictionary/placement counts, union-selected and unsupported-placement counts, maximum
   depth, whole-file row/byte totals and unique candidate `root_placements` (`root_id`, name, count, placement ID).

All references point to definitions already emitted. World is placement zero/depth zero; every other parent is an
earlier placement. A placement's full native path is reconstructed by following parents, not by adding a lossy
path string. `candidate_root_id` identifies only a root at that node; full nested root memberships are derived
from that field along the parent chain. `selected` is the inherited union flag and counts nested unions once.
Local transforms are native object-to-parent mm/rad, not Mu2e-origin-subtracted or invented world matrices.

Whole-file rows are exactly `2 + materials + solids + logical_volumes + placements`. Capture checks this budget
before adding any dictionary/placement record; final serialization checks every actual row again. The independent
one-million-placement hard ceiling remains. Footer bytes are a fixed-width 20-digit decimal string including
itself and the final newline. `unique_paths` is a validated placement count, no longer a global path-set size.

## Solid/material limitations retained

Only exact native `G4Box`, `G4Tubs` and `G4Polycone` entity types have machine-readable max_digits10 parameters.
Polycones retain native original ordered Z-plane data with a 16,384-plane ceiling, not a navigation-equivalent
reconstruction. Const native `StreamInfo` is called once per unique supported solid, bounded at 65,536 bytes;
its presentation precision/units are diagnostic, not the authoritative parameter numbers.

Booleans/displaced/multiunion/custom/other solids remain `unsupported_metadata_only` with null parameters/text
and explicit reason. No recursive unknown-solid StreamInfo or fabricated CSG operands/transforms. Unsupported
counts remain **per expanded placement**, even though solid records are interned. `resolved_all_solids=false`
is mandatory when any placement uses an unsupported solid. Material name/index is not a material-composition
export. Even resolved allowlisted parameters would not prove topology, overlap, containment or navigation.

## Compact frozen snapshot and helper allocation budget

Native traversal runs **once**, capturing placement names/IDs/12 transform doubles and dictionary descriptions
into a snapshot. Definitions are captured once per actual material/solid/LV pointer. No duplicated full ancestry,
full-path set, full serialized placement-row buffer or second native walk/StreamInfo is retained. Two serializer
passes read the same snapshot: first exact byte/row/escaping/precision preflight, then exclusive fd output.
Preflight completes before file creation; the second pass checks final counts before emitting the footer. It does
not reread mutable native geometry or require guessing that two native traversals had identical content.

The snapshot/maps/sibling sets/string buffers use C++17 `std::pmr` through a checked allocator. Requested live
allocation bytes, including old+new buffers during growth and transient serialization copies, are capped at
**min(configured file cap, 256 MiB)**. Nonthrowing move assertions protect nested PMR resource retention during
vector growth. This is a new conservative helper-allocation guard, **not an RSS limit**: native geometry,
allocator metadata, stack, stream/locale internals, ordinary bounded error/context/filename strings and native
StreamInfo internal allocations are outside it. Names remain stored per placement and duplicate temporarily in
sibling checks. Allocation refusal throws; there is no truncation/thinning/partial-success mode.

The successful v2 zero-transport run reports VmHWM 1,695.52 decimal MB and VmPeak 1,868.78 MB, compared with
2,032 MB VmHWM in the earlier failed v1 attempt. This is observed whole-process memory, not an equal-work benchmark,
a general v2 bound or a speed forecast. Worst-case distinct objects can still legitimately exhaust either guard.

## Exclusive publication and acceptance

Filename remains `<prefix>.inventory.run<G4RunID>.pid<PID>.worker<G4ThreadID>.jsonl`; parent directory must exist and
be trusted. One fd, `O_CREAT|O_EXCL`, mode 0600, no reopen/truncate/retry-name/cleanup. Partial writes/EINTR handled,
then fsync and checked close. A failure after creation leaves a rejected partial artifact; flush/close failure may
leave footer-looking bytes. Footer alone is never successful publication evidence.

Readers require valid v2 rows/footer/counts/actual byte length **and** an externally supplied successful closed
writer/job receipt bound to exact inventory/configuration/source/binary/geometry hashes. Structural parse success
without that receipt is diagnostic only. No native file digest or provenance signature is invented; external
hashes identify supplied bytes, not authenticity. Physics/library eligibility stays false regardless.

The Python reader exposes `audit_file(path, limits=Limits(...), receipt=CompletionReceipt(...))`; the receipt's
SHA256, writer-closed and job-success flags must come from independent successful execution evidence. Without it,
valid structure returns `accepted=False`. The report stores primitive placement columns and reconstructs exact
native paths with `report.path(placement_id)`. It rejects unused/reordered first-encounter dictionaries and
inconsistent ordered daughters under repeated logical-volume IDs as well as bad references, roots, selection,
JSON/types/nonfinite values, footer/counts/bytes or caps. Its 136 synthetic tests do not exercise the native writer.

## Completed native structural check and remaining gates

The user's v2 build now checks these C++17 PMR constructions on p110. Fresh source/overlay/binary/linkage checks
passed before module loading and native initialization. The job closes successfully with one rejected art event,
zero G4 events and empty boundary ledger. `recording/audit_inventory.py` binds successful runtime/build receipts,
actual source/binary/config/geometry-entry-file hashes and logged expanded geometry provenance, then supplies an
exact inventory-SHA completion receipt to the independent reader. The audit is structural only, not independent
native store/transform/containment validation. The source-only manifests below retain preparation-time status;
separate post-build and audit receipts record the later execution.

Remaining adversarial native gates include known nonidentity transforms/shared daughters, duplicate siblings,
root store/world uniqueness and nesting, cycles/replicas/nonplacements anywhere, finite values/unsupported solids,
dictionary-induced row exhaustion, exact byte/footer boundaries, helper allocation failure, exclusive collision/
write/fsync/close failure and externally rejected incomplete output. Full-world structural serialization completed;
independent native equality and navigation/containment remain untested. No new G4 event/campaign is approved; original pilot and saved-input controls stay immutable.
