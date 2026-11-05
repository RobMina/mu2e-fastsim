# Native-shaped source diagnostic v2

This schema describes an **unapplied, uncompiled diagnostic proposal**, not a library, a persistent product format or a POT certificate. All envelopes retain `library_eligible:false` / `proposal_only:true`, EventID, row ordinal and existing geometry-file/hash-policy fields. Envelope schema is `target-recording-proposal-v2`. Caps cover source rows as well as native rows and completion. Strict post-read audit remains bounded and always ineligible.

## Source prefix before native event begin

1. `source_begin`: positive generator/output ProductIDs, stage, configured input tag, process-history ID, actual generator branch/module/instance/process/class and nonempty parameter-set IDs, generated-entry and proton counts, row/volume/byte caps, `persistent_output_verified:false`, `exposure_certified:false`. Configured label/instance and an explicit process must agree with resolved provenance; an unspecified process is recorded as resolved, not invented. Branch-name construction, ProductID/history/config authenticity and cross-file registry binding remain external/unverified.
2. `generated_primary`: actual generator ProductID/key, PDG, generator ID, Mu2e position (three doubles, mm), four-momentum (px,py,pz,E, MeV), global/proper time (ns); `generator_weight:null` because the pinned class has no such member. Keys cover the actual collection, including an explicit empty collection. This is not a claim that stored energy/proper time is copied unchanged by the G4 converter.
3. `source_end`: exact entry/proton counts. Native `event_begin` and inventory follow; EventID/helper and caps must match the prepared source. A partial prefix is incomplete, never a zero.

## Links and actual insertion

- `primary_link` immediately precedes each root `stacking`: assigned native ID, actual generator ProductID/key/PDG, native classification and classification-time native state, plus availability of the authoritative input pointer. The draft validates input pointer object identity in memory; no pointer address is serialized. Killed roots keep their link but acquire no fabricated SimParticle.
- `sim_particle_insert` immediately precedes an accepted `track_begin`: actual stored object's product/key/stage/PDG/creation code and parent/generator pointer coordinates, native track/parent IDs, `insertion_observed:true`, `persistent_output_verified:false`. The hook is inside the insertion path, not an inference from a call that may return early. Parent links refer to actual observed insertions; native IDs are not general persistent keys.
- `sim_particle_omitted` records the native size-limit return and prevents a complete source-shaped audit. It is not a synthetic inserted particle or permission to drop the event.
- Other native lifecycle/step/birth/cut rows retain their diagnostic meaning. Earlier exterior branches, same-step boundary production and re-entry/ownership are not solved by identity links.

## Assembled in-memory collection

After all supported track lifecycles, `sim_particle_final` records each actual assembled particle's insertion identity plus daughters. The collection must contain exactly the observed insertions, with unchanged identity/link metadata and reciprocal, unique in-collection parent/daughter coordinates. Output pointers are checked by coordinates and actual collection lookup, not prematurely dereferenced.

`source_reconciliation` follows, with exact generator/proton/link/insertion/omission/final totals and `in_memory_links_consistent:true`. It explicitly retains persistence/exposure false. Only then may `event_end`/`completion` include matching source counts, `source_reconciled_in_memory:true` and `persistent_output_verified:false`.

New hook counters are `prepareEvent`, `simParticleInserted`, `simParticleOmitted`, `reconcileParticles`, including explicit zeros. Existing counters, native abort checks and fatal-cap semantics remain required. A successful structural audit reports only `source_reconciliation:structurally_consistent_in_memory`; source/native/geometry/physics/exposure/library validity all remain false.

## Missing evidence

These rows supply no original written-art verification, authenticated request/shard/SubRun accounting, job/runtime/geometry completeness, cross-file registry binding, normalization or causal launch-frontier algorithm. A separate [uncompiled original-product reader and synthetic structural join](TARGET_READBACK.md) now propose event/SubRun reconciliation; they have not opened an original art file or authenticated its registry/exposure. The [adapter plan](TARGET_SOURCE_ADAPTER.md) and [library contract](TARGET_LIBRARY.md) keep those separate gates. V1 remains supported only as a source-unverified historical diagnostic; v2 rows/counters cannot upgrade it. No build/native/G4 permission follows from this schema.
