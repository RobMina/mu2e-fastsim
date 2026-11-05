# Boundary JSONL diagnostic audit contract

`calofastsim.boundary_ledger.audit` parses the current
`boundary/payload/Mu2eG4BoundaryRecorder.hh` schema
`mu2e-boundary-diagnostic-v1`. It returns a compact `AuditReport`, not transition
rows, entrance-set training data, a response library, or a physical-zero claim.
**The tests are synthetic emit-format tests, not a native execution certificate.**
Nothing here proves hook execution, native transport, geometry coverage, complete
ancestry, feedback closure, raw membership, transforms, or stock equivalence.

## Independent evidence and API

```python
from calofastsim.boundary_ledger import audit, audit_file, LedgerError, TRACK_STATUS_SCHEMA

# manifest is supplied from independent, pre-response source evidence.
# All columns have the same bounded length; integers must remain integers.
report = audit(
    "boundary.jsonl",
    independent_manifest=manifest,
    provenance={
        "purpose": "diagnostic_boundary_ledger",
        "source_sha256": source_sha256,
        "independent_manifest_sha256": independent_manifest_sha256,
        "track_status_schema": TRACK_STATUS_SCHEMA,
    },
)
```

Required manifest columns are `source_sha256`, `run`, `subrun`, `event`,
`root_product_id`, `root_key`. Supply one independently designated generated root
per event. This interface does not accept or reconstruct a multiple-root set.
Do **not** build this manifest from surviving ledger, detector, hit, or CSS rows.
The existing `audit` API retains an **external checksum assertion**: it validates
SHA syntax and column/source consistency, but does not compute the manifest or
source artifact digests. Its report has `manifest_checksum_verified=false` and
`source_checksum_verified=false`; callers must verify artifacts themselves.
Neither API establishes the evidence's independence.

For optional strict artifact verification use:

```python
report = audit_file(
    "boundary.jsonl",
    independent_manifest_path="independent-manifest.json",
    source_artifact="actual-source.art",
    provenance=provenance,  # same required fields as above
    max_manifest_bytes=256 * 1024 * 1024,
    max_source_bytes=256 * 1024 * 1024,
)
```

The manifest artifact is exactly a JSON object with the six columns above, each
stored as a list. Its canonical bytes are
`json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False).encode("ascii")`, **without a final newline**. The strict wrapper
reads at most `max_manifest_bytes + 1` before parsing, rejects noncanonical bytes,
and computes SHA256 from the actual artifact bytes, requiring exact equality to
`independent_manifest_sha256`. It also hashes the actual `source_artifact` in
bounded chunks, enforcing `max_source_bytes` and equality to `source_sha256`.
Every manifest row's source SHA must match that verified source digest. Existing
manifest row/identity limits and reconciliation still apply. Artifact failures
occur before ledger parsing and have no prefix report. Successful verification
sets both checksum flags true, including on a later ledger error's prefix report.
Verification proves byte identity and declared source mapping, **not** that the
native ledger actually came from that source, that manifest roots are correct, or
that the manifest was independently constructed. Those remain external gates.

One invocation binds exactly one source artifact SHA256 to the ledger through
`provenance.source_sha256`. Every manifest SHA must equal it. The native ledger
contains no source SHA or generated-root declaration, so the binding is explicit
external provenance, not something inferred from local G4 IDs. SHA strings are
64 lowercase hexadecimal characters. Cross-source batches require separate
invocations and independent checks of native-ledger/source correspondence. The returned
`ledger_sha256` hashes bytes actually admitted by the parser; after rejection it
may identify only a prefix, never assume it hashes the entire source file.

Input is a path or a binary stream implementing bounded `readline(size)`.
Manifest columns are one-dimensional NumPy arrays or flat lists/tuples. Run,
subrun and ProductID use uint32 ranges; ProductID must be nonzero. Event and keys
retain unsigned 64-bit ranges exactly, including values above the float-exact
range. JSON ID fields must be integer tokens: floats and booleans are rejected.
No conversion through floating point or G4-parent-derived root assignment occurs.

## Strict, bounded protocol

Default limits are 2,000,000 ledger rows, 256 MiB ledger bytes, 1 MiB per line,
and independently 2,000,000 manifest rows. All limits must be positive Python
integers. These are explicit allocation/work bounds, not a precise RSS guarantee;
choose smaller limits in restricted environments. Manifest lengths are checked
before coercion, key construction, sorting, or output allocation. Nested manifest
lists, object/float/bool integer arrays, invalid hashes and duplicate source events
are rejected. JSON is decoded only after bounded binary reads. A single-byte
lookahead detects row/byte excess; oversized lines are rejected before decoding
or JSON object allocation. The ledger and source artifact are never read wholesale; the optional canonical
manifest is materialized only within its separate explicit byte cap.

Each line must end with a newline and be exactly one JSON object. Blank lines,
unknown/missing fields, duplicate keys at any nesting level, NaN/Infinity,
overflowing float values, overlong integer tokens, nonfinite state, and malformed
identity/state types are hard errors, not row filters. Accepted state numbers may
be integer or floating tokens because the native emitter prints integral doubles
without a decimal point. Nested state, touchable, material and persistent-identity
objects are validated. World-null is allowed only for a world-boundary step;
material metadata need not be null there.

Event blocks cannot interleave or repeat identities. `event_begin` has sequence
zero; every later sequence is contiguous and all run/subrun/event IDs must match.
`event_end` must match G4 event ID and `rows_before_end`. Begin conventions must
be candidate-only, world-frame, `mm_MeV_ns`, and art uint32 ProductID encoding;
candidate root names must be nonempty and unique. Touchable membership is
recomputed from these names as a consistency check, **not geometry certification**.

Track starts, born-inside observations, entries, exits, unresolved births,
tracking ends and resumes are checked serially as a streaming protocol. Identity
and saved-at-start flags cannot change. Initial births must have step zero;
entries, exits and unavailable-membership transitions must have positive steps.
An unknown birth's first-observed-pre row must be at exactly step one. Steps
cannot go backwards; transitions cannot repeat within one step. Visit IDs start at zero (outside/unknown) or one
(born inside), increment only on entry or first observed inside unresolved birth,
and must match every subsequent row. An exit must close an active visit; a
terminal-inside end closes its visit without inventing an exit.

Provenance must explicitly declare
`track_status_schema="geant4-11.3.2-p110-G4TrackStatus-v1"` (the exported
`TRACK_STATUS_SCHEMA`). This fixes installed p110 Geant4 11.3.2 semantics:
Alive=0, StopButAlive=1, StopAndKill=2, KillTrackAndSecondaries=3, Suspend=4,
SuspendAndWait=5, PostponeToNextEvent=6. Values outside 0–6 are invalid;
**5 is not postponement**. Birth statuses are 0 or 1; tracking-end status 0 is
invalid. Status 1 (at-rest lifecycle) is retained as unsupported, not interpreted
as suspend. Resume requires a preceding nonterminal end with status 4 or 5,
then an Alive=0 resume row at the same step, with unchanged visit and touchable
membership consistent with the saved end membership and active visit. Statuses
1 and 6 cannot resume in this supported event-local protocol. Terminal statuses
2 and 3 can never resume. Nonterminal ends retain the
visit until resumed/terminated; unresolved or postponed tracks remain incomplete.
The fatal native `membership_unavailable` row can be the final diagnostic prefix,
never a completed event; any following row is rejected.

End counters for unknown births, missing saved particles, unknown ends and active
visits must equal observed lifecycle counts. Overflow gives status 10; otherwise
step/field kills give status 1; otherwise status 0. Aborted events are failed even
when this status is zero. Stock-filter rejection is retained independently and
is not reinterpreted as transport failure. Track counts below observed starts are
rejected; a larger native track count is retained only as native metadata, not a
proof that every possible track was recorded.

## Reports, errors and reconciliation

`AuditReport.events` contains bounded per-observed-event columns, not full rows.
Root columns come **only** from matched manifest rows. `manifest_matched=false`
marks foreign events, whose root columns are zero sentinels, not assigned roots.
The principal fields are:

- `ended`, `status`, `transport_failed`, `stock_filter_pass`;
- observed `starts`, `entries`, `born_inside` counts, unknown/missing counters, `unclosed_visits`,
  `open_tracking`, and `pending_tracks` (not terminally finished);
- `incomplete`, `unsupported`, `postponed`, `no_entry_observed`,
  `domain_occupancy_observed`, `no_domain_occupancy_observed`;
- `library_eligible`, always false.

`no_entry_observed` says only that no exterior entry row was seen. It includes
born-inside cases and failed/partial events. `born_inside` counts are exposed per
event and as a total in `counts`. Born-inside observations remain retained and
are **unsupported for the current exterior kernel**, even after a closed terminal
end. `no_domain_occupancy_observed` is distinct: it is true only when no accepted
birth/first-observation/transition/end/resume membership observed the domain.
It is observational, not a complete-coverage claim; unknown or partial histories
can still have this flag true. Neither flag is **no particles**, no
persisted response, no deposited energy, or successful zero. Unknown birth
membership remains unsupported even if later observation locates the particle.
Missing saved identities, unknown tracking ends, nonzero simulation stages and
postponement and at-rest status 1 are explicit unsupported diagnostics. Nonterminal unfinished tracks
remain incomplete even when outside and no visits are active. A failed event can
have a complete protocol; always inspect transport failure separately.

All reconciliation is vectorized using structured exact-integer event keys and
NumPy sorting/search, not per-event analysis loops. The necessarily serial
operation is stream parsing and lifecycle state validation. Event order need not
match manifest order. `manifest_indices` maps observed events to independent
manifest rows (`-1` for foreign), while `missing_manifest_indices` and
`foreign_event_indices` identify omissions and additions. Missing and foreign
events yield `reconciled=false`, never disappear through an intersection join.
Duplicate ledger events are a hard protocol error; duplicate manifest events
are rejected before streaming. Empty ledger plus a nonempty manifest reports all
missing; empty ledger plus empty manifest is structurally reconciled but certifies
nothing about native execution or physical zeros.

EOF at a line boundary inside an event returns a retained incomplete event and
`protocol_complete=false`. Malformed JSON, a partial final line, invalid ordering,
inconsistent counters and exceeded resource caps raise `LedgerError`. Its
`report` retains observed valid event beginnings and independently reconciles
that prefix, with `protocol_complete=false`. Invalid provenance/manifest errors
occur before streaming and have no report. A malformed first begin cannot supply
an accepted event identity; it remains absent/missing in the prefix report.
Counts of raw rows/bytes can include a rejected admitted line, while event columns
represent diagnostic prefix state, not a transactionally accepted dataset.

Callers must check both `reconciled` and `protocol_complete`, and separately
inspect incomplete/unsupported/failed flags. None of these flags promotes a
library, constructs ancestry roles, or certifies transport success. Persistent
parent and generator pointers are syntax-checked only. Local G4 parent numbers
are never persistent lineage evidence, and a first entry is not classified as an
independent exterior ancestor. No nearest-state matching, response-based
selection, per-incident draws, event splitting, or training export is provided.
The broader production gates in `ENTRANCE_SET.md` still apply.
