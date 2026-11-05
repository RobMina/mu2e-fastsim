# Native target-recording source proposal

**Unapplied, uncompiled, experimental diagnostic only; always library-ineligible.** These files are not an installed Offline package. They do not modify defaults or the running simulation. Do not apply to an arbitrary release or request a production run from this directory.

## Assets and base

- `target_recording.patch`: null-gated hook/configuration and source-list changes against Offline commit `b24d4163e5e8f5ed2b5f72fccb25e4980fec64e2`.
- `TargetRecording.hh` / `TargetRecording.cc`: proposed sidecar diagnostic sink, copied into `Mu2eG4/inc` and `Mu2eG4/src` only in a new review tree.
- `source_pins.json`: exact reviewed base and proposal hashes; unrelated source changes are not silently accepted.
- `beamfastsim.tsrecordproposal`: hash-checks and assembles a new temporary review tree without changing the base or proposal, compiling, constructing geometry or executing transport.

The hook path is absent by default because `TargetRecording` is an optional typed table. When provided, it requires ordinary `GenParticles` input and internal G4 filtering off. MT use is explicitly refused. No recording defaults are added to production FCL. The source list gains the new helper in CMake; SCons already discovers ordinary `.cc` files through `make_mainlib`. The CMake main-library list explicitly names art Framework_Principal, canvas and fhiclcpp for source/provenance access; the pinned SConscript already lists these direct dependencies, so its content stays unchanged and hash-bound. This still requires compile/link verification.

The proposed A/B and in-memory rows are described in the [source schema](../TARGET_SOURCE_SCHEMA.md) and [source/POT adapter plan](../TARGET_SOURCE_ADAPTER.md).

A separate [original-product readback proposal](../TARGET_READBACK.md) comprises `TargetReadback_module.cc`, `readback_pins.json` and `beamfastsim.tsrecordreadback`. It is not installed or compiled by the recording patch/assembler. Its synthetic structural join does not authenticate original art, provenance, successful framework termination or POT. No actual-art readback has been performed with it.

## Hooks and noninterference intent

- Source preparation in `Mu2eG4::produce` exclusively reserves the event directory before transport, reads the configured generator handle and provenance, and writes every input entry plus a generated-proton count. Native event begin must match that prepared EventID/helper; it then inventories geometry. Unbegun or failed transport leaves an incomplete directory.
- Stacking records the actual short-circuit decision and binds a secondary birth token to the assigned native ID even for killed tracks. Primary rows use the authoritative primary-helper input Ptr, checked against the configured handle including actual object identity; no addresses are persisted. A stacking kill has no fabricated SimParticle.
- A hook inside the actual SimParticle-start path records insertion with the stored object's key/stage/parent/generator links; the size-limit return has a distinct omission row. Begin tracking requires one such outcome. End tracking observes native terminal state before the existing end-state helper.
- Before recorder completion, the assembled in-memory collection must match inserted identities and reciprocal parent/daughter links, and every generator entry needs one primary classification. This is not written-product verification or qualified POT.
- Stepping observes paired native endpoints and secondary production before common stepping cuts. Separate cut-result records observe the actual kill/short-circuit outcome. Existing cut evaluations and branch order are retained.
- Event end observes `G4Event::IsAborted()` separately from Mu2e status/limits, refuses an aborted event, and closes the sidecar after SD handoff. The marker means **recorder end-hook and ledger completion**, not proof that the enclosing framework callback returned, native run termination succeeded or art persisted products. Original raw art, generated source/SubRun counters, final job logs, configuration/runtime hashes and post-write reconciliation are separate required evidence.

The sink must not call a second navigator, change track/touchable/step state, draw random numbers or modify weights/physics/cuts. Disabled-path noninterference is a source-review intent, not a measured runtime result. File I/O, exceptions and memory changes on the enabled path can affect runtime and must be bounded and independently checked.

## Eligibility and known gates

The [recording contract](../TARGET_RECORDING.md) remains authoritative. The sidecar is not a source library, a populated production table, an independent POT denominator, a certified union-boundary definition or a transport closure result. No causal launch-frontier implementation is supplied. Target selectors must come from reviewed native inventory and exact expected material; names alone cannot qualify geometry. Internal target-to-target transitions, world escape, zero-step tracks and birth/transfer incidence require explicit interpretation, not momentum-based normal inference.

Native pointer-to-birth-token mapping is transient only. Assigned track IDs must never be inferred from secondary order, time or species. Missing associations, surviving pending births, repeated lifecycle states and unsupported continuation are completeness failures, not zero-exit events. A `fKillTrackAndSecondaries` end with unassigned offspring is explicitly refused; no deleted-child disposition or assigned ID is fabricated. Upstream 11.3.2 source shows deletion without stacking on that path, but the installed manager implementation has not been byte-matched or executed. Native installed status/header APIs and callback order are source inputs, not executed coverage. Early kills can prevent downstream lifecycle callbacks and must reconcile with stacking decisions and raw art.

Post-step material is recorded independently of post-touchable placement material: their equality is not assumed at a boundary; only the pre-step material is checked against its placement. Actual installed transportation callback semantics remain unqualified.

`beamfastsim.tsrecordledger.audit` checks bounded JSONL callback structure, native-shaped IDs/membership links, lifecycle/birth conservation, counters, caps and post-read file hashes using synthetic fixtures. The new v2 stream adds generated-input payload, authoritative primary-link, insertion and final in-memory reciprocal-link checks through `tsrecordsource`. V1 stays a source-unverified historical diagnostic and cannot carry v2 rows or be upgraded by adding counters. It rejects missing/duplicate stacking, nonterminal/suspended ends, unresolved births and overflow. It does not qualify uninterpreted physics payload fields, actual source products, full exterior steps, native geometry or POT; successful receipts keep all validity/eligibility flags false.

All geometry membership, callback ordering, secondary-birth identity, persistence, native status/field/cut effects, finite-limit behavior and disabled/enabled controls remain unvalidated until separately reviewed, user-built and explicitly approved tests. A no-native Python patch-assembly result establishes only exact source bindings and applied text. It cannot detect all C++ API/link issues or prove Geant4 semantics.

## Source-only review command

In an environment providing the Python package, choose a **new, nonexistent** destination outside both input trees. The strict text applicator requires exact hunk offsets and context; it never invokes git or accepts fuzzy matches:

```bash
PYTHONPATH=beam python3 -m beamfastsim.tsrecordproposal \
  --source-root /path/to/pinned/Offline \
  --proposal-root beam/native \
  --destination /path/to/new/review-tree \
  --out /path/to/assembly-receipt.json
```

This assembles only the reviewed base-file subset, not a complete muse area. It cannot be built as a full Offline checkout. Preserve a failed tree and its marker for diagnosis rather than overwriting it. No build or simulation command is embedded in the tool. A complete isolated source area and explicit user-run build are later steps, not implicit permission.
