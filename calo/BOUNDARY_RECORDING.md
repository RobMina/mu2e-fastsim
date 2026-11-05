# Proposed calorimeter boundary and membership record

Status: design/acceptance authority with partial default-off diagnostics assembled and user-built in the isolated
current-head area. Module loading, bounded generator-only manifests and zero-transport structural inventory checks
have passed; native boundary transitions, containment/material composition and raw membership off/on invariance
have not. No new transport is approved by this document. See `PAUSE_RESUME.md` and `recording/current_head/README.md`.
The existing-art numeric exporter cannot invent missing transition histories.
See `recording/INTERFACE.md` for material ownership and exterior-feedback requirements and
`recording/ENTRANCE_SET.md` for the separate diagnostic entrance-set container, not a populated library.

## Scope and decision

The target is a response domain containing both calorimeter disks, their front/back/side passive material,
FEBs and the cable structures explicitly assigned to the calorimeter. Specify that domain by versioned native
volume/touchable identities, not by the name of a VD or a fitted acceptance cut. Resolve whether remote cable
runs and surrounding structures belong to this domain or to upstream transport. A disk/FEB shell union is not
a sufficient definition: cable material can occupy space outside those shells.

Use existing vacuum mother volumes where their union encloses the intended domain. Before selecting them,
export actual placed solids, transforms, parent/daughter identities and material assignments and check the
containment of every included material volume. Do not add a material-bearing mother or overlapping sensitive
shells merely to simplify recording. If the domain cannot be represented by existing mother subtrees, the
boundary recorder must evaluate the versioned union using native pre/post touchables and navigation evidence.
A geometry-only validation and a reviewed source patch precede any new bounded transport request.

## Transition ledger

Record actual domain membership changes from the stepping action or an equivalent native hook; a generic
sensitive detector that emits every step does not distinguish entry from another step within a thin shell.
For each transition retain:

- source artifact/run/subrun/event and persistent product/key particle and designated generated-root identity;
- G4 track ID and step number as local evidence, never substitutes for persistent identities;
- ordered transition index, visit ID, surface/domain version, entry/exit and first-entry/reentry classifications;
- pre/post step status, full pre/post touchable paths including replica/copy numbers, and material identifiers;
- double pre/post position, momentum, energy, global/proper time, step length and terminating process;
- the crossing endpoint and an explicit convention for its time/state, not a linearly extrapolated VD center;
- parent identity, creation position/time/process, and a born-inside record when the track begins in the domain.

An inbound transition is an outside-to-inside membership change, not merely a momentum sign. Outward normal
and momentum are useful validation columns. Treat tangencies, shared surfaces, repeated boundary steps,
zero-length navigation, born-on-surface ambiguity and aborts explicitly. An exit/reentry of a previously entering
lineage belongs to the same source event. Record branch creation and escape histories needed to separate
upstream siblings from secondaries produced by an already entering shower.

## Independent event manifest and correlated conditioning

Write one event-manifest row before any detector-response filter, with zero-length offsets permitted. Reconcile
it against finished input EventAuxiliary identities and transport status. A missing product is not an empty
response, and failed/aborted transport stays in a failure ledger. Preserve the original source-artifact checksum,
source-pool subset policy, generated identity, geometry/material/configuration hashes and runtime provenance.

Each independent event owns an ordered set of exterior entrance lineages. Store energy, direction, position,
time and species for every member, with lineage and transition references. The state set is measured at the
chosen transport boundary; do not condition a response model on hidden true deposits. Preserve correlations
between independent co-entering siblings and deposits/leakage in both disks. Do not train independent photon
samples from one root event, or choose one entrance according to which produced a hit. A truly single upstream
root boundary is an alternative only if it precedes relevant branching and encloses the proposed response domain;
its upstream transport and conditioning support must then be stated and validated.

Do not populate the existing single-entry schema with one arbitrary member of a correlated entrance set.
Version any eventual multi-entry schema separately. Include events with no exterior entrance, born-inside cases,
unsupported topology and out-of-support states in explicit ledgers rather than deleting them. A completed event
with no persisted stock crystal response is a threshold-defined zero only after its incident manifest is certified.

## Raw-to-compressed membership

At the stock compression producer, emit an optional association from each accepted raw crystal StepPointMC
(product ID + ordered source index) to its actual output CaloShowerStep row. Record the chosen persistent
ancestor, crystal, depth bucket, time-group generation, flush cause and each constituent's order. Raw rows
rejected by the stock energy threshold get a reason record. Preserve exact configured collection order and
compression parameters, including their FHiCL storage types. The diagnostic must not change sort comparators,
accumulation, thresholds, producer ordering, random streams or normal output selection.

Keep equal-time stock sort order as observed provenance. An offline replay can reproduce counts/scalars within
an equal-time domain without proving which indistinguishable raw rows stock assigned or in which summation order.
Do not create associations by matching average time, position or fit values. Source-algorithm replay remains a
control until compared against independently emitted membership.

## Acceptance gates before a library claim

1. Native geometry containment and explicit boundary coverage, including phi/radial surfaces and cable gaps.
2. Exact independent event identities/status and registry reconciliation, including empty and failed events.
3. Complete first-entry, born-inside, escape and reentry ledgers; no response-conditioned incident selection.
4. Root-coherent correlated entrance sets and a documented causal assignment, including multiple disks.
5. Independently observed raw-to-compressed associations and source-field precision checks.
6. Same-step stock response/digitization/reconstruction controls, then independent held-out CE and cosmic checks.

A build is performed by the user after source review. Runtime packaging and a new transport campaign require
separate authorization. Existing raw/compressed products can still be used for deterministic response controls
while these boundary gates remain closed.
