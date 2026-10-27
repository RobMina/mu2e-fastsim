# Opt-in Cat v2 extraction and event census

`crvfastsim.veto.catsteps_v2` prepares census-bound Cat tables and `PoolV2` for a
future versioned replay. It does **not** change `catsteps`, `mixframes`, replay,
response, timing, selection, BO objectives or existing files. The intended future
primary target is **CE-conditioned**; this change only repairs input identity and
denominator accounting. It does not implement the CE-conditioned intensity draw,
certify factorisation, or turn the legacy unconditional mixture into that target.

## Required extraction

One successful dumper job must process **all events from exactly one source art
file**, with no event selection, skip, maximum-event truncation or concatenated
source files. The new `CrvStepTableDumper` source adds `dump/events`, with one row
per successfully analyzed event:

- `run`, `subrun`, `event`: native unsigned art identifiers;
- `ngen`: unsigned 64-bit generator count, including zero;
- `nsteps`: signed 64-bit step count, including zero; **-1** if `writeSteps=false`.

The events tree is created and filled even for generator-only extraction. Such a
job is a useful generator census, but cannot become a Cat step pool: v2 requires
step extraction and rejects negative counts or missing trees. Failed jobs/partial
dumps must not be presented as successful full-source extractions. The converter
checks internal consistency; it cannot prove that a caller's declaration of
full-source processing is true without an independent source census/job record.

Required trees are `dump/events`, `dump/gen`, `dump/steps`, `dump/particles`, using
the highest ROOT cycles. Old dumps lacking an events tree are rejected, not upgraded
by inferring events from generator or step rows. Missing empty-generator events
cannot be recovered that way. Empty step and particle trees must retain their
branch schemas. Subrun zero is valid; run/event zero and reserved maximum unsigned
IDs are not. No bit-packed keys are used.

## Explicit provenance and conversion

Prepare a JSON file with these fields (replace every placeholder):

```json
{
  "source_art": {
    "identity": "catalog-qualified-unique-art-file-identity",
    "uri": "/absolute/original/source.art",
    "sha256": "SOURCE_ART_SHA256"
  },
  "extraction": {
    "id": "unique-extraction-attempt-id",
    "command": "exact invoked extraction command with source and output locators",
    "software_version": "frozen release/area identity",
    "fcl_sha256": "FROZEN_EFFECTIVE_FCL_SHA256",
    "dumper_sha256": "DUMPER_SOURCE_SHA256",
    "write_steps": true,
    "source_count": 1,
    "event_selection": "all"
  }
}
```

Hash fields require lowercase, 64-character SHA256 strings. `source_art.uri` must
be an absolute path or a URI, not a staging basename such as `dump.root`. The
catalog identity, original source content hash and extraction information are
**caller declarations**, to be produced and retained by the extraction workflow;
the converter does not fetch/open the original art source or verify an executable.
The dumper source hash is not proof that the corresponding binary was built.
Use one frozen effective FHiCL for compatible pool members; do not put per-file
command-line source names into that configuration hash.

In an analysis-only Python environment with this checkout and core on `PYTHONPATH`:

```bash
python3 -m crvfastsim.veto.catsteps_v2 "$FULL_DUMP" \
  --dataset "$CAT_DATASET" --provenance "$EXTRACTION_JSON" \
  --out "$NEW_V2_PARQUET"
```

Python dependencies: NumPy, pandas, uproot, PyArrow; pytest is needed for the
synthetic tests. Conversion writes two files: the requested step Parquet and a
sibling named by replacing its suffix with `.events.parquet` (for example,
`cat.parquet` and `cat.events.parquet`). **Neither destination may already exist**,
including an orphan companion or dangling symlink. Both are created exclusively;
ordinary write failures clean up only newly created outputs. Publication of the
pair is not crash-atomic: a crash can leave an incomplete pair, which must be
reviewed/removed before retrying, never silently overwritten. Retain and move the
two files together; the companion filename is recorded relative to the steps.
The converter returns the compact metadata object as well as writing the pair.
It hashes the actual local ROOT dump before and after reading, saves its absolute
locator/hash, hashes its own source, and preserves original source/extraction
declarations. No remote I/O, source submission, grid job, response or build is
launched by this CLI.

## Validation and persisted schema

Parquet metadata key `crvfastsim` contains `schema: "crv-cat-v2"`, dataset, counts,
source art and extraction provenance, dump and converter hashes, and a compact
`events` reference containing only `path` (the companion filename) and `sha256`
(the hash of its actual bytes). There is **no embedded `event_census` JSON**:
metadata and manifests do not grow with the number of census events. This remains
a preparation schema, `crv-cat-v2`; no external schema or real Cat data has shipped.

The `.events.parquet` companion contains exactly the columns
`run`, `subrun`, `event`, `ngen`, `nsteps`, all int64, with rows sorted by full event
tuple, including stepless and empty-generator events. Loading rejects missing companions, byte-hash mismatches,
wrong columns/types, invalid IDs/counts, and duplicate events. The reference must
be a relative sibling filename ending in `.events.parquet`, not an absolute path,
URI, directory traversal, symlink, or self-reference. The file hash is checked
before and after reading. `census_sha256` independently binds canonical typed,
sorted census rows, while `content_sha256` binds that census together with compact
steps using the recorded `arrow-ipc-stream-v2` format. Updating a companion's file
hash alone cannot conceal changed census content. Content hashing sorts numeric
step rows, so row reordering does not evade duplicate-content checks;
source identity is deliberately excluded from this content hash. Hash buffers
canonicalize all NaN signs/payloads as nulls with zero-filled data slots and an
explicit validity bitmap (including zero padding), and both signed zeros as +0.
This affects hashing only: stored kinematics are not rewritten, NaN times remain
missing, and every finite nonzero value and numeric dtype remains significant.
Pandas indices and metadata are excluded. The earlier `arrow-ipc-stream-v1`
encoding included noncanonical null-slot bytes and can fail after a valid Parquet
roundtrip. It is rejected: reconvert the original ROOT dump with the current
converter to a **new output pair**, retaining the old pair/provenance for diagnosis;
do not patch hashes or disable validation. Arrow IPC is an
implementation-versioned content encoding, not a universal physics equivalence
hash; use a compatible PyArrow environment. The pool also records each actual
Parquet file hash separately. These are consistency checks, not signed attestations.

Step rows contain `source_identity`, full `(run, subrun, event)`, particle key and
PDG, and the existing compact kinematics. Spatial/energy/momentum fields remain
float32; times are float64. NaN times are preserved for downstream reviewed timing
handling, but infinite times and nonfinite/overflowing kinematics are rejected.

Conversion rejects:

- missing/noninteger/out-of-range event IDs, or duplicate census IDs;
- any step, generator or particle event missing from the census;
- disagreement between generator/step row counts and census `ngen`/`nsteps`,
  including the per-step `nstep` counter when provided;
- duplicate generator indices or particle keys, even with equal PDGs;
- unknown/invalid particle keys or PDG zero, inconsistent **step/particle** PDGs,
  or nonexistent generator keys when nonnegative `genidx` is present;
- missing particle matches (there is no unknown-PDG fill or silent join expansion).

A particle's PDG need **not** equal its linked generator's PDG. Offline's
`Compression/CompressDetStepMCs_module.cc` can attach an ancestor's generator
pointer to a retained compressed descendant of another species. `genidx >= 0`
therefore requires event-scoped generator-key membership only; `-1` denotes no
link. Generator and particle PDGs must still be nonzero valid integers, and each
step must match its own particle's PDG when a step PDG is supplied.

Particle identity within an input is `(run, subrun, event, key)`. Frame identity
in a pool is **`(source_identity, run, subrun, event)`**. Equal numeric event tuples
from genuinely distinct sources remain distinct. Generator indices are not frame
identities, and generator multiplicity never determines the denominator.

## PoolV2 contract

```python
from crvfastsim.veto.catsteps_v2 import PoolV2
pool = PoolV2(files)
```

The structural interface matches the legacy pool: `.frames` includes every census
event, `.stepped` counts stepped frames, `.steps` is a grouped pandas DataFrame,
and `.start`/`.count` are int64 arrays indexing each stepped frame's contiguous
rows. `.ids` is a tuple-valued MultiIndex; `.steps.fid` is only a dense local group
index, never a persisted identity or bit packing. `.events` retains the complete
source-qualified census. A pool with positive census size and **zero steps** is
valid and has empty offsets/counts with a usable typed step table. An empty file
list or zero-event census is rejected.

Pool loading rechecks schema, provenance, full identity, counts and content hashes.
Members must agree on dataset, software/configuration/dumper/converter versions and
full-extraction declarations. Different per-file commands, source locators and
extraction attempt IDs are expected. `.manifest` retains paths, actual file hashes,
source/dump/content hashes and all compact metadata, including the companion
filename/file hash and canonical census hash (not the census rows). The manifest
shape is unchanged; `.hash_groups` exposes memberships
for `file_sha256`, `dump_sha256`, `source_sha256` and `content_sha256`.

**Duplicates fail closed.** Repeated source identity, source locator, extraction
attempt, source hash, dump hash, Parquet hash or normalized compact content hash
is rejected. Nothing is silently deduplicated, merged, weighted or renormalized.
This deliberately also rejects independently declared sources with identical
compact content (including identical all-empty samples): ambiguity requires a
separate provenance/weighting decision, not an automatic override. Disjoint names
and distinct hashes do not establish independent underlying showers. This version
has no duplicate-acceptance policy.

Future replay may explicitly choose this pool interface, but no existing driver
selects it automatically. New identity/denominator contracts must be retained in
that replay's provenance; this preparation alone is not a rate or response closure
result and does not authorize changing production defaults.

## Build and validation status

The C++ census edit is **unbuilt**. No art extraction, remote execution, grid work
or detector-response test was performed for this change. Existing libraries do
not establish that the new events tree exists. There are no new C++ dependencies,
so no SConscript/CMake changes are required.

The user must run the full build on **mu2ebuild02**, in the intended `art/` muse
satellite area, then confirm completion before any art smoke extraction:

```bash
cd "$CRVFS_CODE/art" && source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && setup muse && muse setup && muse build -j 32
```

After build confirmation, a separately authorized smoke extraction must verify
one census row per analyzed event, empty-generator/stepless coverage and the
`writeSteps=false` sentinel. Python tests are **synthetic only**:

```bash
python3 -m pytest -q tests/test_veto_catsteps_v2.py
```

They cover full-tuple collisions, empty events/pools, count and join failures,
provenance/hash disagreements, valid compressed-descendant generator links,
duplicate-file/source/content refusal, companion tampering/missing/orphan/path
protections, no-overwrite/write-failure cleanup, and compact metadata for a
100,000-event mostly-empty census, plus signed/payload-bearing NaNs, signed zeros,
nondefault pandas indices, deterministic null buffers, and exact finite-value
sensitivity across Parquet roundtrips. They do not validate a compiled dumper, real source completeness, finite-library exposure,
CE conditioning or any physics result.
