# Original written-product readback proposal

**Source-only, uninstalled and uncompiled; always library-ineligible.** `native/TargetReadback_module.cc` proposes a separate sequential art EDAnalyzer over the original written art products. No original file has been opened with it, no module installed, no geometry constructed and no Geant4 run. This is not an EventNtuple, gallery or regenerated-product substitute. `native/readback_pins.json` binds this proposal separately from the existing recording patch; the recording manifest/assembly do not install or compile the reader.

## Reader boundary

Four literal, explicit-process InputTags select the actual `GenParticleCollection`, `SimParticleCollection`, `StatusG4` and SubRun `GenEventCount`. Missing handles, unresolved or wrong-product pointers, configured/resolved provenance disagreement and duplicate EventIDs fail. Actual map-vector keys and stored object IDs are kept separately. Parent/generator/daughter pointers are checked for ProductID/key, availability and resolved object identity against the loaded products. `links_resolved_in_readback:true` means resolution, not graph consistency or authenticated provenance. Empty collections are serialized, not replaced with one proton.

The reader writes bounded JSONL into a new exclusive directory. It closes `readback.pending.jsonl` successfully before exclusively publishing the `readback.jsonl` hard link; the pending name is retained as evidence. A partial/failed close has no final name. Publication is not fsync/crash durability or successful enclosing framework termination. No destructor emits completion. Hard-link support and callbacks remain unqualified native requirements.

One input file is active at a time; reopened names, repeated EventIDs/SubRuns, cross-file SubRuns, events after SubRun closure and input-close with outstanding SubRuns are refused. A SubRun callback count must equal observed written EventIDs in that SubRun. Counts are module callbacks, **not protons or completed transport**. Initial support intentionally refuses split/continued/empty SubRuns. Installed headers establish callback signatures and `FileBlock::fileName()`, not their runtime ordering. If art closes the final file after `endJob`, or delivers endSubRun outside an active file, this proposal will refuse completion; do not silently assign a filename.

Particle caps apply independently to each loaded collection and each daughter list. Row and byte caps include header, file, SubRun and terminal rows; the event cap counts event rows only. Product caps apply after art deserialization, not to ROOT/framework memory. No output-selection or weight-product policy is inferred.

## JSONL schema

Every row has exactly `schema:"target-written-products-proposal-v1"`, `kind`, `data`, `proposal_only:true`, `library_eligible:false`; each row ends in LF. Order:

1. `reader_begin`: `generator_tag`, `sim_tag`, `status_tag`, `count_tag`, positive `max_rows`, `max_bytes`, `max_events`, `max_particles_per_event`.
2. `file_open`: actual `input_file_name` from the file callback.
3. `event`: `input_file_name`, `event_id:[run,subrun,event]`, `process_history_id`, generator/sim/status ProductIDs and six-field provenance, `generated_primaries`, `sim_particles`, `status`, `links_resolved_in_readback:true`, `persistent_output_verified:false`.
   - Provenance: `branch_name`, `module_label`, `product_instance_name`, `process_name`, `friendly_class_name`, nonempty unique `parameter_set_ids`.
   - Generator entries reproduce v2 `generated_primary` fields exactly, including actual dense collection key, finite position/four-momentum/times and `generator_weight:null`.
   - SimParticles: `product_id`, `key`, `object_key`, `sim_stage`, `pdg`, `creation_code`, `parent`, `generator`, `daughters`; a pointer is null or `{product_id,key}`. No native track ID is reconstructed.
   - Status: `status_native`, `sim_overflow`, `step_kills`, `field_kills`, `n_g4_tracks`. The last is a separate readback observation, not equated to SimParticle count.
4. `subrun`: `input_file_name`, `run`, `subrun`, `count_product_id`, `count_provenance`, `generated_event_callbacks`.
5. `file_close`: matching `input_file_name`; all that file's SubRuns closed. Multiple file groups must be disjoint.
6. `reader_end`: exact `event_count`, `subrun_count`, `file_count`, `readback_callback_complete:true`, and false `framework_job_success_verified`, `persistent_output_verified`, `exposure_certified`.

## Bounded structural join

`beamfastsim.tsrecordreadback.audit(readback_directory, native_directories, requested_event_ids=...)` requires complete strict v2 native-shaped ledgers and exact requested/native/written EventID sets. It re-hashes both native files on the join pass, matches every generated input entry and original-product-shaped SimParticle identity/parent/daughter set against assembled in-memory finals, checks object keys and reciprocal links, and compares status observations. Abnormal status, overflow or limit kills refuse the intended shard. A stacking-killed root has a generator link but no fabricated SimParticle. Generated proton counts are observations; completed-proton count and POT normalization stay null.

Source and readback process-history IDs are retained **separately**: framework processing can append history. No equality requirement, authenticated relationship or history-content claim is fabricated. Caller row/file/event/particle caps and an aggregate native-byte cap bound retained evidence. All authenticity, persistence, exposure, geometry, native and physics certification flags remain false, even for synthetic streams that match perfectly.

Synthetic controls are defined in [tests/test_tsrecordreadback.py](tests/test_tsrecordreadback.py), including payload/schema, identity/link, EventID/SubRun/file closure, status, cap and serialized-byte controls. They do not qualify the compiler, art callback ordering, native Ptr resolution or original-file readback. Test-source presence is not an executed-test receipt; the [checkpoint guide](UPSTREAM_CHECKPOINT.md) states the recorded test scope separately.

## Still required

- Hash-bind original file bytes and real product/branch registry, process history/configuration and output selection/drop rules; strings and a filename are not authentication.
- Qualify reader callback ordering, Ptr resolution, source/runtime freshness, flush/publication failure behavior and actual original-art readback after a separately user-approved build.
- Bind requested/generated/begun/completed/written sets to an independently checked request and complete producer/reader termination logs, not caller-provided JSON alone.
- Review generator/event weights and actual proton multiplicity policy; deduplicate original shard identity, restart/overlap and copied SubRun counters; enforce development/held-out source disjointness.
- Implement causal launch ownership, native geometry/boundary controls and transport physics closure. This reader does not produce launch states.

No build command or readback/transport execution is authorized by this document. A standalone source file and a successful synthetic join cannot establish an unbiased target-exit library.
