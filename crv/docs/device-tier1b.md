# Opt-in tier-1b device sampling

The supported reference sets are strict_v3 (context, contain, point, topology,
unvisited, near-miss, chord and showers) and strict_v4 (the same plus hard-muon
draws). The extracted model selects strict_v4; NumPy remains the backend default.
Mirror, translate, follow and stop-product sampling are not part of this port.
The reference sampler and asset sets are not modified. Device range_out transport
is an independent opt-in twin of the extracted transport rules.

## Sampling contract

Main donors are uniform over stable (group/momentum/path bin, context) support,
falling back to the bin with fewer than 20 context donors. Zero-history donors
remain in support. All steps of a selected history are retained; unvisited
rows are excluded from the main component and sampled separately. Chord
mapping preserves the fraction inside the donor chord and distance outside.
Other-bar placement tries bar structure, point repair, then topology override;
nearest same-axis fallback, midpoint inset and endpoint clipping match NumPy.

Unvisited support uses the finest of five position levels with at least 20
donors, otherwise the coarsest nonempty level. Translation is admitted only
in the original plane and outside the receiving muon's visited planes.
Near-miss selection excludes entire events with any fast crossing. Every
remaining generated muon draws reach by its straight-ray cell and, on reach,
a uniform complete template. Nonmuon showers draw reach by absolute PDG and
kinetic energy, then uniform complete shower. Absolute template positions and
generation-relative times are preserved. Descendant identities include source
and crossing; shower primary/root keys use the reference 10^7 namespace.

Preparation builds immutable sorted support and CSR histories from the checked
library/sidecars. Every support bound and history length comes from the whole
library, never a validation shard. Missing positive-probability support rejects;
there is no history cap or silent truncation. All geometric arithmetic is
float64. Separate JAX keys provide independent draws; fixed-uniform replay
compares deterministic selection and placement against NumPy.

## Strict-v4 hard draws

Library preparation uses `attach_hard` without changing its threshold or donors.
Each fast muon's ordered group/layer pattern is encoded as two base17 integer
halves, with the momentum class of its first crossing. Exact tuple lookup gives
the key catalogue and hard rate; sparse keys use the class/count rate. More than
24 crossings have zero hard rate. Accepted hard muons take one whole matching
catalogue history; other crossings with a supported hard key redraw within their
original support while the donor is hard, for at most 50 rounds as the reference
does. Residual hard picks are reported, not silently repaired. The two per-muon
uniform arrays and compact active-redraw arrays preserve fixed-uniform replay.
Scalar counts drive dispatch; patterns, picks and row expansion stay on device.

## Current implementation boundary

See [measured results and failed gates](device-tier1b-performance.md).

Experimental, not yet certified. Library arrays, crossing metadata, donor
selection, CSR expansion and main/unvisited placement live on the selected JAX
device. `device_metadata` accepts padded crossing mappings with explicit valid
lanes. `plan_descendants` / `plan_unvisited` retain donor picks and complete
history prefix sums. `device_blocks` yields bounded JAX placement buffers with
event keys, relative identity/source/parent, PDG, times, energy and path fields.
No fixed per-history cap is used; additional blocks drain the entire history.
Only scalar row counts and missing-support status control host dispatch.

The existing DataFrame API collects these blocks for schema/ancestry publication.
Shower assembly, near-miss candidate grouping and final ancestry remain on the
host. Device generation, device transport and response interfaces also retain host-table boundaries. Therefore
this is not a zero-copy or fully device-resident chain claim. Compilation,
library preparation, numeric execution/transfers and publication must be
reported separately from steady-state end-to-end timing.
