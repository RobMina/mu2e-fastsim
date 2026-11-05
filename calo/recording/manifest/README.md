# Independent incident manifest — REVIEW SOURCE ONLY

**The corrected pass-through filter implementation awaits a user rebuild and native tests.**
A built analyzer draft was rejected by art when scheduled on a trigger path; moving it to an end path would violate
the intended pre-response ordering. This directory holds reviewed source, not a completed input manifest.
The original pilot and its saved-input controls are unchanged.

## Placement and meaning

Two always-pass art EDFilters use the existing CaloLibraryDumper TFileService/ROOT idiom:

1. `CaloInputEventManifest`, the **first module** on the single trigger path,
   records `(ordinal, run, subrun, event)` before TargetStopResampler, `generate`,
   counters, filters, Mu2eG4, or response. It consumes **no products**.
2. `CaloIncidentManifest` immediately follows `generate` and precedes genCounter,
   GenFilter and Mu2eG4. It explicitly declares/reads only the designated
   `mu2e::StageParticleCollection` via `generate::CaloLibraryCE`. It never searches
   all products, dereferences ancestry, or consumes transport/response products.

The example fragment replaces PrimaryPath after including the unfiltered pilot.
The trigger path, not an end path or response-selected path, is essential. Audit
expanded FHiCL for all alternate paths, services, exception policies and early
modules before integration. Use one thread and one schedule; ordinals describe
actual callback order, not sorted EventIDs. Existing output stays on its end path.
A resampler's rejection skips the incident phase but not the initial ledger.
Successful callbacks always return true; failures still throw after attempted rows. Both writers belong under
`physics.filters`. The plugin kind and TriggerResults bookkeeping change; no framework-output identity is claimed.

**Source identity precedes generation:** configured `sourceArtifact` and its SHA
identify the immutable upstream stop-pool bytes supplied to TargetStopResampler,
not the generated output art file and not an identity inferred from response.
This pilot uses EmptyEvent-assigned primary EventIDs; they are **not** stop-pool
EventIDs or sampled entry numbers. The StageParticle's parent ProductID/key is a
numeric art link into the resampled product, not proof of a stop-pool file entry.
Mapping random resampler choices back to original pool entries needs independent
resampler provenance instrumentation and is outside this proposal. For RootInput
integration, instead name/hash the actual source artifact and externally enumerate
its accepted source order. Multiple input files require an audited canonical
artifact inventory; this example only covers one immutable pool.

## Output contract (ROOT, no custom dictionary)

TFileService module directories `inputEventManifest/` and `incidentManifest/`
contain:

- `metadata`: schema 1, source logical ID, configured source SHA256, snapshot
  manifest SHA256 and normalized expanded FHiCL SHA256; incident tag/type/mode,
  explicit `coordinate_frame=Mu2e` and native mm/MeV/ns units.
- `inputEvents`: one row for every source EventID that reaches the first manifest filter,
  including events whose generation later fails.
- `events`: one row for every event reaching the incident filter, even a valid
  **empty** collection; present/count/product ID/`single_root_supported` flag.
- `products`: the designated collection registry only, with actual provenance
  `branchName()`, explicit tag/type, numeric ProductID, presence and count. Missing
  products have empty branch and ProductID 0 and are not mislabeled empty.
- `roots`: every native collection element in original index order. Identity is
  `(source SHA, EventID, ordinal, kind, ProductID, index)`; ProductID alone is not a
  file-global identity. `kind=2` means StageParticle; `kind=1` is reserved for
  GenParticle and is never emitted here. No GenParticle fallback is implemented.
  Stage creation code is **not** GenParticle generator ID. Parent validity,
  numeric parent ProductID/key are retained without dereference. `pdg`, native
  creation process, double position/momentum/time are copied directly from
  StageParticle; no float narrowing, transport start state, or derived mass.
  Units/frame follow native Mu2e StageParticle (mm, MeV, ns, Mu2e coordinates);
  no geometry transform is performed. This is **not `G4_world`**: a future ROOT
  adapter must apply the reviewed Mu2e-to-G4_world transform before downstream
  use, not relabel these coordinates. StageParticle has no native proper time.

`rootMode=record_all` emits zero/one/multiple roots without dropping events and
sets `single_root_supported` only for one root. `require_one` emits all rows then
throws on zero or multiple roots. A missing designated product always throws
**after** the event/registry row attempt. Neither mode selects one of many roots.
Rows attempted before an exception are not a durability guarantee.

## Generator zero/exception semantics and limits

The pinned CeEndpoint source creates exactly one StageParticle for a suitable
stopped muon, and throws BADINPUT if the stopped-muon list is empty. It does not
normally put an empty StageParticle collection. Other generators may validly
produce empty collections, which this design records. A generator throw happens
**before** the incident filter: the initial ledger was attempted, but there is
no root/event row for that callback. An exception skipped by framework policy
thus leaves a sequence gap and the completed-job audit fails. Do not interpret a
missing incident row as an empty event or reconstruct it from output products.

No module callback can be promised for source failures before Event dispatch,
failed beginJob, process crashes, fatal source read failures, or killed jobs.
ROOT buffering/exception shutdown can also lose already attempted writes. This
is not a write-ahead/crash-safe journal. Complete coverage requires an externally
enumerated source EventID sequence, successful job/close receipt and audit; an
interrupted output is diagnostic only, not a complete manifest. The validator
intentionally does not repair gaps or accept a partial prefix as complete.

## Hashes, snapshots, and external validation

`source_manifest.json` pins copied **review source/API snapshots** under `snapshot/`
and proposal files. Relative logical origins are used; no site paths are needed.
StageParticle/GenParticle accessors and CeEndpoint behavior come from those local
Offline source copies. The dumper snapshot pins the TFileService idiom, not the
actual installed art/ROOT ABI. Snapshots are review evidence, **not runtime build
pins**, not a guarantee that the future build matches them. Do not claim full
Offline/art compatibility before a separately reviewed clean build. Upstream
copies retain their original notices; the copied Offline LICENSE governs its
source snapshots. Review copies do not establish an installed runtime identity.

All hash fields are configured claims. The module validates lowercase SHA syntax
but **cannot self-prove that source bytes were actually read**, the runtime's build
identity, or authenticity of a launcher. An independent launcher must:

1. Verify snapshot/proposal pins, hash immutable upstream artifact before execution,
   and record its logical ID, resolved inventory and receipt. Bind that same
   artifact to `TargetStopResampler.fileNames` (no untracked siblings/substitution).
2. Replace review tokens and expand FHiCL. For the FHiCL digest, normalize exactly
   the two actual unquoted `expandedFclSha256` keys’ double-quoted values to
   `<NORMALIZED_FCL_SHA256>`, hash UTF-8 bytes, insert that digest into both fields,
   re-expand and check the normalization hash is unchanged. This explicit scheme
   skips comments/quoted text and rejects lookalike-only inputs. This scheme
   avoids a cryptographic self-reference; it is **not** the raw file SHA. Also
   record the raw final expanded-file SHA externally. Expansion format differences
   change the hash; use the exact executed file. Do not re-expand at runtime.
3. Pin actual runtime/build/plugin/source/config identities and process success in
   an independent receipt. Check the approved path order, incident tag/type, thread
   settings, source bounds, exception policy and stop-pool binding externally.
4. Independently enumerate the exact primary EventID list in execution order. For
   EmptyEvent this comes from the approved run/subrun/first-event/count contract;
   for RootInput use independent source enumeration, with configured skips/limits.
   Never derive this list from downstream art retention or either manifest tree.
5. Close ROOT successfully, rehash source bytes (immutable inputs recommended),
   compare both metadata rows and all event/collection/root sequences using:

   ```sh
   python3 validate_review.py check-snapshot
   python3 validate_review.py normalized-fcl-hash approved.expanded.fcl
   python3 validate_review.py audit --manifest incident_manifest.root \
     --artifact upstream.art --expanded-fcl approved.expanded.fcl \
     --expected-events expected_event_ids.json
   ```

`expected_event_ids.json` is an array of `[run, subrun, event]` triples with
JSON integers in [0, 4294967295]; floats, booleans and strings are rejected. Audit
requires NumPy and uproot and exact TFileService labels from the example. It rejects gaps,
reordering, missing/extra roots, numeric identity/type mismatch, count/support
mismatch, nonfinite state, non-double ROOT state branches and external hash
mismatch. ROOT trees are read once as NumPy columns; event/root comparisons
use vectorized arrays, without row dictionaries or per-event loops. Zero and multiple roots are valid in `record_all`. This is structural
completeness validation, not state-by-state independent generator replay, proof
of cryptographic authenticity, or proof of absence of other upstream selection.

## Ready-to-integrate footprint and checks

- `src/CaloInputEventManifest_module.cc`, `src/CaloIncidentManifest_module.cc`,
  `src/ManifestMetadata.hh`: two pass-through filter plugins and local metadata helper.
- `CMakeLists.txt`, mirrored `src/SConscript`: proposed libraries only; no build invocation. The corrected filter API must still be compile-checked.
- `fcl/manifest_ce_review.fcl`: deliberately invalid review tokens until prepared.
- `validate_review.py`, `tests/test_manifest_source.py`, snapshots and hash manifest.

Source/pure-data tests only, using the project's Python venv:

```sh
python3 -m pytest -q -p no:cacheprovider calo/recording/manifest/tests
```

These tests inspect schedule/consumption/types/pins and exercise external hashing,
empty/multiple/strict cases and sequence-corruption rejection. They do **not**
compile C++, load a plugin, expand FHiCL, run art or G4, produce/read ROOT fixtures,
or validate TFileService exception persistence. Before promotion: compile against
pinned Offline/art; expand the actual production config; run small one/zero/many,
missing-product, resampler-reject and generator-exception fixtures; verify ROOT
branch types and persistence, plus full independent source enumeration. No build
or production integration is authorized by this review directory.
