# Entrance-set diagnostic schema 1

This is **not** library schema 2, a trained response model, or a physical-zero certificate.
`calofastsim.recording.pack` retains source events with arbitrary correlated incident multiplicity, both-disk
showers, empty recordings, failed transport and incomplete recording. It always emits `library_eligible=false`.
There is no function to promote that flag. External provenance declarations cannot bypass a production gate.

## Tables

- **Independent manifest:** source artifact SHA256, source shard, run/subrun/event and designated generated-root
  ProductID/key. Write this before any detector-response selection and reconcile against finished input records.
  A source event is the indivisible draw; this version supports one designated generated root per event and must
  reject rather than split a source requiring a root set. Multiple incident particles from that root are allowed.
- **Events:** the exact manifest identities plus transport status, recording-complete and topology-supported flags.
  These are diagnostic flags supplied by independent audits; the packer does not prove them. Nonzero-status and
  incomplete events are retained and flagged, not silently converted into successful empty showers.
- **Incidents:** event index, observed sequence, persistent particle ProductID/key, positive visit ID, species,
  role and floating state `(x,y,z,px,py,pz,kinetic_energy,time,proper_time)`. The incident table contains exterior
  incoming crossings, returns and born-inside observations, not every transition. The complete native transition
  and ancestry ledgers remain referenced evidence; they are not recoverable from this table.
- **Steps:** event index, source CSS ProductID/index, crystal, SimParticle ProductID/key, constituent count and
  stored state `(x,y,z,time,energy_g4,energy_birks,momentum_in)`. Keep original floating dtypes and persistent
  identities; do not infer associations from averaged step state.
- **Provenance:** diagnostic purpose plus domain, geometry, expanded-FHiCL and independent-manifest checksums.
  These identify evidence; the packer cannot validate its physics content. Separate source/runtime/patch and
  raw-membership manifests belong alongside this recording.

Role values are `0=initial exterior candidate`, `1=return`, `2=born inside`, `3=unclassified`. A role is an audited
lineage/visit classification, never inferred by nearest time/position or by whether the particle made a hit.
Different upstream siblings remain correlated in one source event; a returning descendant is not an independent
initial input. First entry of a new particle identity can still be feedback from an escaping ancestor.

## Exact state conventions and batch bound

Provenance must state `coordinate_frame=G4_world`, `units=mm_MeV_ns`, and
`product_id_encoding=art_ProductID_value_uint32`. Positions and momentum components use the unshifted native
world frame; times are native global/proper ns, momentum components MeV/c, kinetic energy MeV. A CSS exporter
using Mu2e or crystal-local coordinates requires an explicit transform to this frame; copying its numbers without
that transform is invalid. Preserve source coordinate/transform hashes alongside the recording. The packer cannot
infer or verify the physical transform merely from this declaration.

ProductIDs are the integer returned by `art::ProductID::value()`, from 1 through 2^32-1. Particle keys and source
indices retain separate nonnegative int64 storage; never reinterpret a G4 track number or textual `process:product`
identity as that integer. Manifest `root_product_id/root_key` names the designated generated particle; incident
and step SimParticle references name their respective SimParticle products. An eventual native-text adapter must
resolve registry identities and test its round trip; no such adapter is supplied here. Born-inside observations
at rest are valid; a classified incoming/return crossing requires nonzero momentum.

The packer accepts at most `max_rows` total event+manifest+incident+CSS rows per batch (default 2,000,000), checked
before integer copies, structured-key sorting and output allocation. This is a row bound, not a precise RSS bound;
choose a lower bound for restricted memory. Process one bounded source artifact or whole-event batch at a time,
never split a source event to satisfy the cap. Artifact-to-shard consistency must also be audited across batches.

## Invariants

Source artifact, shard and whole-event split are one-to-one. Every independent manifest event is retained exactly
once, including no-entry/no-response and failed events. Incident sequence and particle/visit keys are unique within
each event. CSS product/index keys are unique within an event. Missing identities, unresolved event references,
nonfinite state, malformed integer IDs and negative deposited energy are hard errors, not row filters. Incident
rows are ordered by event and observed sequence; CSS rows preserve observed order within each event. Offsets may
repeat. No crystal-level split or independent per-incident draw is constructed.

`initial_exterior_candidate` is only a role selector, not training eligibility. It cannot certify native material
coverage, ancestry/visit classification, entry-state sufficiency or exterior-feedback closure. Stored zero-length
step ranges mean **no persisted CSS**, not literally zero Geant4 energy. Those cases remain subject to the same
threshold and independent-manifest audits as positive responses.

## Deliberately absent

No interpolation, nearest-neighbor retrieval, conditioning on true deposits, response-based incident selection,
physical-zero promotion, out-of-support substitution or stock injection is implemented. Schema 1 is a container
for prospective recording/audit, not a reclassification of existing pilot VD rows. Native recordings must first
pass geometry, complete lifecycle, independent source/status and stock raw-membership gates in INTERFACE.md.
