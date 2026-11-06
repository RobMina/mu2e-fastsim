# Causal photon response reference

## Scope and prerequisites

This stage builds on [the inclusive photon reference](flash-inclusive.md): frozen
recording, spatial first-foil extraction, matched primary-brems checks, and the
complete input-event denominator. It adds causal timing, exclusive ownership,
saved-response comparison and residual ancestry diagnostics. It does not replace
the inclusive reference's timing/normalization, TS3, geometry or generalization
requirements. In particular, raw tracker response is not historical board closure,
and no fitted normalization can supply missing non-target contributions.

## Recorded boundary information

The all-photon step record supplies pre/post positions and momenta, pre-step
GlobalTime, step length and end process. These support reconstructed spatial
crossings, not native G4 boundary tuples. It has no post-step time or unique
post-volume identifier. Photon-cut coordinates are Mu2e coordinates; raw straw
positions are tracker coordinates. Birth volume comes from SimParticle stage
and start-volume index joined to the SubRun volume table, not the step copy number.

## Crossing times and causal frontier

`flash_g4_timing` brackets an accepted spatial crossing between its raw pre-time
and the next same-photon step's pre-time. A two-segment reconstruction requires
the step after the continuation. A defined particle end-time rounding interval
provides a fallback upper bound; asymmetric float32 rounding bounds are retained.
No chord/c diagnostic is promoted to an exact time.

`flash_g4_frontier` keeps every birth row but marks a descendant exposure covered
only when an ancestor crossed strictly before the immediate child on that branch
was born. An ancestor that exits after emitting that branch cannot own its future
response. The rootmost eligible ancestor wins; overlapping time intervals,
non-exiting foil ancestors, missing pointers and depth exhaustion remain unknown.

## Exclusive response and accounting

For additive response, each tracker deposit must have at most one owning
interface photon. A photon and its later descendants cannot both own the same
deposit. Reentries and descendants born before an ancestor crosses require an
explicit causal assignment; a nearest-photon birth attribution is not by itself
a boundary-response definition. Start from the complete interface exposure and
left-join attributed deposits, filling absent contributions with zero. Keep
unassigned/non-target energy as an explicit residual.

`flash_g4_response` exclusively attributes raw tracker ionization to this frontier,
left-joining zeros. It preserves definite non-interface residual and unresolved
energy separately, with per-event accounting. Missing products, status failures,
collection caps, unknown exposure metadata and ambiguous deposits prevent known
zero-response labels. Recording checks are not a proof that every transport or
recorder failure was signalled; logs still need inspection. This remains a
first-foil, geometry-dependent raw-response diagnostic, not a fitted universal
response or the historical board objective. Resolved-only subsets must not be
renormalized into an inclusive response.

## Paired response comparison

`flash_response_compare` reads the saved response NPZs, requiring identical full
input-event identities and ordinals. It reports paired event and SubRun-cluster
uncertainties, event concentration, and coarse energy-binned response on the
explicitly resolved subset. Bidirectional energy-only predictions propagate
response/exposure/observed-response covariance at event level. They are diagnostics
on development anchors, not design holdout. Empty energy bins are unsupported;
observed-zero bins are unvalidated plug-in zeros, even if their empirical standard
error is zero. Neither a small pull with sparse showers nor good rank agreement
establishes transferability.

## Saved-shard merging

A saved-only `flash_response_merge` combines only event/photon comparison columns
after count,
recording and duplicate-identity checks; raw tracker-owner row indices are local
to their original shard and must not be naively concatenated.

## Residual ancestry census

`flash_response_origins` reads saved tracker-active event ancestry to explain the
residual without changing it. It distinguishes nearest and earliest observed
photons, source-stage continuation particles and full roots. Negative foil-birth
ancestry requires a complete chain and stage/volume metadata. A particle can pass
through the target without being born there, so this census alone cannot certify
independence from target geometry. Non-target-produced photons and charged
contributions must be accounted for separately rather than absorbed into a fitted
target-photon normalization. The larger reference can include non-photon branches
born in target foils, for example nuclear-capture protons descended from an input
negative pion. A residual is not synonymous with upstream background or with a
geometry-independent contribution, even if two residual totals happen to agree.

`flash_response_origins_merge` combines the compact saved censuses without joining
shard-local row pointers. It checks full tracker identities, sidecar event/step
counts and all per-event energy categories against the independently merged event
ledger. Source-species/foil-ancestry categories use event-summed sampling errors,
not independent-hit errors. Counts describe the supplied ancestry archives; this
is not a fresh raw-export completeness test.
