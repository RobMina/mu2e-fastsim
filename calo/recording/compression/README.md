# Optional observed compression-membership patch (unapplied data)

**Review proposal only.** This directory contains a unified patch, exact inspected source copies,
a SHA256 manifest and source-level tests and explicit opt-in resource caps. It does not install or modify Offline, enable recording,
build software, run art, or authorize another transport campaign. Existing compression replay is
still algorithmic evidence until compared with independently emitted membership.

The inspected tree is a **paused source snapshot, not the actual pinned runtime build**. Patch
applicability is not build readiness, ABI/API compatibility, a runtime test, or validation of a
production release. User review, an independently authorized build, source/library/configuration
pinning and a same-input stock-vs-instrumented control remain necessary.

## Contents and exact applicability

- `membership.patch`: changes only `CaloMC/src/CaloShowerStepMaker_module.cc`, relative to an
  Offline source root (`patch -p1` layout). It has not been applied to Offline.
- `snapshot/`: byte copies of the producer, `ShowerStepUtil.hh/.cc`, CMake and SConscript inspected
  for this proposal. They are context/test data, not an alternative build tree.
- `source_manifest.json`: SHA256 for all five source/build files and hashes of the two design
  documents read during preparation. Context-document hashes are historical evidence, not a
  requirement that unrelated documentation never change.
- `check_applicability.py`: verifies every source/build hash, then dry-runs and applies the patch
  **only in a temporary copy inside this directory**, rejecting fuzz/offsets. It never writes to
  the supplied source root. The OS `patch` executable is required.
- `test_membership_patch.py`: pytest-collectable `unittest` tests, requiring no C++ toolchain; including preallocation row checks, bounded text insertions and prewrite byte accounting.

The producer's exact inspected SHA256 is:

```text
18ad4ccf05fcd31fdeed362d3485e681d1f97e31f500ee91fd18775626fbbf2a
```

Run from this directory, supplying a source root explicitly:

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/project/.venv/bin/python3 check_applicability.py /path/to/Offline
PYTHONDONTWRITEBYTECODE=1 /path/to/project/.venv/bin/python3 -m pytest -q -p no:cacheprovider test_membership_patch.py
# Equivalent source-only checks when pytest is unavailable:
PYTHONDONTWRITEBYTECODE=1 /path/to/project/.venv/bin/python3 -m unittest discover -s . -p test_membership_patch.py -v
```

For this delivered patch, SHA256 values are:

```text
membership.patch: 03d754d3d519e72f43372a97b04a9bc1180537d9bb1fdeb0e405b4a5bcc5e05b
patched producer: d41a0800a0b9b4166d7a68779ee6269953b592e1a4f783f3e93c0e3051d800ec
```

An exact snapshot hash match and zero-fuzz temporary-copy application passed locally. All 14
source-level tests passed with the project virtualenv pytest command above. Tests check source
structure, cap enforcement sites and disabled-path invariants, **not C++ execution semantics**.
The cap edge cases still require an independently authorized runtime/build test. No compilation, linking, runtime job, remote execution, staging or commit
was performed. Changes to the inspected source require a fresh review, not a relaxed hash check.

## Activation and dependency decision

The proposed typed configuration adds an optional path and two positive limits:

```fcl
# Empty string is the default: no diagnostic file or diagnostic source registry.
membershipLedger: ""
membershipMaxRawRowsPerEvent: 2000000
membershipMaxBytesPerJob: 268435456 # 256 MiB, including the header
# To opt in after review, choose a unique, non-existing file for one producer/job:
# membershipLedger: "compression-membership.txt"
```

The path declaration is `fhicl::Atom<std::string>` with default `""`. Both caps are
`fhicl::Atom<std::size_t>` (platform unsigned size type); zero is rejected before file creation
when recording is enabled. Caps do not activate diagnostics on their own. A nonempty path enables an
explicit text ledger independent of TFileService. The producer opens it in exclusive-create
`"wx"` mode, never appends or truncates an existing file, and throws on open/write/flush failure.
The parent directory must already exist. Use a different filename for every job/process/module
instance; no concurrent writers or shared-file merge are supported. C-library support for this
exclusive mode must be checked in the eventual runtime. The design relies on the serialization
of this legacy `art::EDProducer`; no threaded/replicated runtime has been certified.

An `art::Assns` product was not selected: the full ordered constituents, rejection records and
flush metadata would still need an additional representation and dictionary/schema review. This
proposal adds no art product or dictionary, no ROOT service, and only standard-library headers.
Consequently **both** inspected `CaloMC/CMakeLists.txt` and `CaloMC/src/SConscript` stay unchanged;
there are no new external dependencies to mirror between them. `ShowerStepUtil` is also untouched.
If the eventual reviewed implementation uses TFileService or Assns instead, re-review its schema,
dictionaries and mirrored build dependencies rather than treating this patch as coverage for it.

## Hard limits and bounded diagnostics

Before creating any raw-address registry or running stock grouping, the recorder sums the sizes
of **all** resolved handles using `size > max - total` checks, rejecting more than
`membershipMaxRawRowsPerEvent` rows without overflow. Exact equality is allowed; empty handles
count zero. Missing handles still fail at stock lookup. Shadow membership/generation tables use
sparse bucket maps rather than allocating diagnostic arrays of `numZSlices` entries, so their
allocation follows observed raw membership rather than an arbitrarily large bucket configuration.
The caps bound diagnostic data, not the input products or stock compression/geometry allocations.

A custom bounded `std::streambuf` replaces the unbounded event `ostringstream`. Every insertion
checks the remaining text budget **before** appending, including characters, quoted tags, every
member/rejection, and bucket/group records. The stream has `badbit|failbit` exceptions enabled;
a cap breach cannot silently truncate a successful event. The header uses that same bounded buffer.
Each event gets only `max job bytes - bytes already written` of text budget. `ledgerWrite` checks
and reserves the entire header/event byte count before `fwrite`; no overflow bytes reach the file.
The cap includes all header/newline bytes and all events; it is per producer/file/job instance,
not a combined multi-process quota. An exact byte boundary is allowed. No full-event `.str()`
copy is made when writing. The bounded string may retain allocator capacity from a prior event;
container overhead/allocator growth means these are row and payload caps, **not** an exact RSS cap.

Exceeding either cap deliberately fails the diagnostic job; nothing is sampled, silently dropped,
or rolled over. A byte-limit failure while formatting leaves no partial current event on disk;
an I/O failure may leave one. In all cases discard any completeness claim for that failed job.
A very small positive byte cap can reject the header itself, leaving an exclusively created empty
file; it never authorizes exceeding the limit. Event blocks remain deferred until the original
`event.put` calls succeed. The header, however, is written during construction.

## Where membership is observed

For each event, the diagnostic first enumerates every configured raw handle in configured order,
including empty handles, and maps **the actual element address** to `(collection slot, ProductID,
zero-based source index)`. Source indices are vector indices, not particle keys or inferred hit IDs.
Repeated resolved ProductIDs throw before stock grouping, even for two empty handles. Different
ProductIDs exposing the same element address also throw. Thus configuring the same collection
twice, including through different tags, cannot silently conflate pointers. These failures are
opt-in diagnostic failures; the default-off stock behavior is not changed.

The original stock `std::sort` comparator and all original buffer additions remain untouched.
Immediately after each successful `buffer.add`, the raw source address is appended to that
bucket's shadow list. Immediately after each actual `caloShowerStepMCs.emplace_back`, the ledger
records `size()-1` and that bucket's ordered source addresses, before the stock reset. The final
bucket loop is instrumented at its actual emissions too. Equal-time order is therefore the order
**observed in that invocation**, not a stable-sort prescription, a replay guess, or matching on
stored average values. Different builds may still choose different equal-time order.

A generation counter starts at zero independently in each depth bucket of each `compressSteps`
call and increments after a time-gap flush. Its full scope is `(event, group, output SimParticle
ProductID/key, bucket, generation)`, not just `(event, bucket, generation)`. This matters with
`compressData: false`: stock still performs ancestor/crystal thresholding first, then calls
`compressSteps` separately for each direct source SimParticle. `G` retains the selected ancestor;
`S` retains the SimParticle actually passed to the output constructor. They may differ.

Every raw row must receive exactly one `A` or `R` disposition. The emitted shadow-list length
must equal the stock buffer entry count. Unknown/repeated/missing source addresses fail the
instrumented event. Diagnostics do not insert a new rejection or alter output selection to make
their ledger pass.

## Text format, version 1

Records are whitespace-separated. Only the configured tag is quoted, using `std::quoted` escaping
for quotes and backslashes; use a compatible quoted-string parser. Integer fields are decimal.
Floating fields are C++ hexadecimal floating-point text, suitable for `float.fromhex` when finite;
no precision-losing decimal formatting is used. Nonfinite input/stock arithmetic is not repaired
or reclassified by this patch. All ProductIDs are art's numeric `.value()` representation.

| Record | Fields following record type |
| --- | --- |
| `H` | `calo-membership 1` |
| `P` | configuration name, FHiCL storage type, effective value |
| `E` | run, subrun, event, actual output CaloShowerStep ProductID, configured collection count |
| `C` | collection slot, resolved raw ProductID, source vector size, quoted configured tag |
| `D` | `zSliceSize double`, actual computed slice size used in this event |
| `G` | group ID, selected ancestor ProductID, selected ancestor key, crystal, sequential group energy sum, group raw count |
| `S` | actual output index, group ID, actual output SimParticle ProductID, key, crystal, bucket, generation, flush cause, member count |
| `A` | actual output index, constituent order, collection slot, raw ProductID, raw source index |
| `R` | group ID, collection slot, raw ProductID, raw source index, rejection-group order, reason |
| `Z` | raw count, accepted count, rejected count, actual output count, group count |

`P` records preserve the original FHiCL storage types: `numZSlices unsigned`, `deltaTime float`,
`compressData bool`, `eDepThreshold double`. Additional `P` records report both effective caps
as `membershipMaxRawRowsPerEvent size_t` and `membershipMaxBytesPerJob size_t`. The float `deltaTime` is printed after the same exact
promotion into the stock double member. `C` records preserve configured collection order/tags,
not a ProductID sort. `D` exposes the stock geometry-dependent derived size including its epsilon.
The expanded configuration, geometry and provenance must still be retained independently.

All indices and slots are zero-based; group IDs start at one per event. An accepted relation is:

```text
(artifact, run, subrun, event, raw ProductID, raw index)
  -> (same event, E.output_ProductID, A.output_index, A.constituent_order)
```

`A` rows follow their `S` row in actual bucket accumulation order. `S` rows occur in actual output
vector order, including time-gap emissions interleaved across buckets and final flushes in stock
bucket order. Flush causes are exactly `time_gap` and `final_bucket`. They are observations of
which emission branch executed, not a fitted interpretation of output time.

`G` is the stock ancestor/crystal group **before** any noncompress subdivision. Its energy is the
existing sequential sum used by the unchanged `eDep < eDepThreshold_` test. Rejection reason is
exactly `below_ancestor_crystal_threshold`; it is a **group** threshold, not a per-step threshold.
`R` order is the actual pre-sort group traversal, not a nonexistent output constituent position.
Rejected rows have no output index, bucket or time generation assigned by stock, so none is
invented. Equality to threshold remains accepted; no positivity/finite guard is added.

Empty events have `E`, all configured `C` records (possibly zero), `D` and a zero-count `Z`; no
accepted or rejected rows are invented. A missing handle still fails via stock `getValidHandle`;
it is not an empty collection and receives no success block. Multiple collections are addressed
by their ProductID/index and configuration slot independently, even when their raw values match.

## Local API evidence (not compilation)

Read-only inspection found the same no-instance `event.getProductID<Collection>()` idiom in
`Filters/src/EMFBoxHitsFilter_module.cc:203` and
`Filters/src/CompressStepPointMCs_module.cc:193,197`, used before output insertion. This supports
the API spelling, but neither proves the `const art::Event&` use in `ledgerBegin` nor certifies the
proposed `CaloShowerStepCollection` instantiation against a pinned art runtime. The local art
`Event.h` implementation was not available in the inspected source. **Compilation remains
unverified**, including the typed limits and stream implementation; do not launch on this evidence.

## Completion, provenance and remaining gates

The event block is assembled in bounded memory and written only after both original `event.put` calls.
Each write is flushed and checked. `Z` reconciles the in-memory block; it is **not** proof that an
art output file was closed successfully or that later path modules accepted the event. The ledger
has no job-completion footer. An interrupted/failed write can leave a partial block or even a
complete-looking suffix; any failed job/diagnostic invalidates completeness claims. Events failing
before `produce`, handle lookup, geometry access, stock compression or diagnostic validation need
an independent failure/event manifest. This recorder is not that manifest or a transport-status
ledger, and it does not certify response zeros or incident coverage.

ProductIDs/event IDs alone do not identify the source artifact globally. Retain the input/output
artifact SHA256, process/product registry, module label, expanded FHiCL, geometry/conditions,
patched source hash, linked library/runtime hashes and job success receipt beside this file.
Join to persisted products only with those artifact and registry identities; do not merge by bare
run/event or match outputs by time, position, energy or ancestry alone. Art output selection can
subsequently drop events/products; reconcile it independently against these producer-local records.

The default-off patch introduces no normal products, output reordering, changed comparator,
accumulator, threshold, selected-particle set or random calls. The source tests guard those narrow
facts; they do not establish bitwise equivalence. Opt-in recording adds memory and I/O costs and
can deliberately throw for invalid diagnostic setup. A reviewed user-built control must compare
normal product fields and order on the same saved inputs with diagnostics off/on, reconcile all
raw dispositions and output indices, and exercise equal-time ties, time-gap/final flushes, both
compression modes, threshold equality/rejections, multiple/empty collections and duplicate tags.
Only that execution can establish scoped runtime correctness. Native boundary transitions,
independent incident/transport manifests, causal entry assignment and a certified shower library
remain outside this optional compression-membership patch.
