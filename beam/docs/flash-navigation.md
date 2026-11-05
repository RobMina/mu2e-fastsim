# Finite-foil navigation: evidence and transaction gates

This is a design contract, not an implemented finite-volume transport API. Existing charged-source transport is
unchanged. The standalone foil ray, helix, frame, competition and local-leg query modules are prerequisites;
photon births are not target-exit flux.

## Pause checkpoint — 2026-11-05

R5 is paused. Photon births and geometry helpers are not validated exit flux; S2,
the 66-point comparison and tracker/CRV occupancy validation remain unfinished.
No owned Geant4 pilot or production-slot use has occurred. The separate pidfd-free
controller has 114 mocked tests but no host launch; its outer watchdog and
invocation-bound receipt are unimplemented. The latest combined focused software
checkpoint was 2029 passed / 10 skipped (overlapping earlier suites, not additive
or a full physics validation). See [the handoff](r5-pause-20261105.md),
[output/lifecycle gates](flash-reference-output-bound.md) and
[owned-child design](flash-reference-owned-children.md). Nothing resumes automatically.

## Three different capabilities

1. **Exact represented surface continuation.** `after_surface` names one already-consumed exact transverse root.
   Forward ownership is checked from the current trajectory. The solver excludes only that root, certifies departure,
   and still searches for later crossings of the same surface. An arbitrary near-surface point is not an exact root.
2. **Numerical geometric evidence.** An isolated curved crossing has an enclosing path interval, a representative
   distance, a monotonicity margin and resolved other-face clearances. Binding that evidence to the original
   trajectory, unique foil and retained frame can identify a geometric event without making its representative point
   an exact boundary state. Guarded floating-point bounds are not outward-rounded interval proofs.
3. **Material-transport restart.** Scattering or energy loss changes the trajectory. A geometric receipt's forward
   limit does not establish material ownership at its approximate representative point. General approximate receipts
   must not authorize a changed-trajectory restart; an interval-state or independently justified event treatment is
   still needed.

Do not conflate these capabilities into one `valid` mask. Capturing numerical evidence alone does not implement
all-foil competition, slab integration or a physical boundary restart. The host-only same-trajectory consumer below
uses verified evidence but still does not supply any of those transport capabilities.

## Implemented host-only evidence emitter

`flash_foil_receipt.bind_inputs` copies the exact supplied numeric trajectory and geometry values, dtypes and shapes,
plus an explicit nonnegative foil ID and optional finite numeric `(4,4)` frame binding. The frame binding is opaque:
it is not inferred, transformed, validated as a proper frame or treated as proof of external provenance. A repeated
foil ID within a batch must retain identical geometry. Global identity still belongs to the caller's foil table.

`first_transition_with_receipts` returns the unchanged four-array helix result and a flat C-order tuple of immutable
receipts or `None` for each broadcast lane. The private NumPy observer captures supported isolated crossings after
the ordinary hit arithmetic. It adds nothing to the default JAX graph. Zero-force, no-hit, unresolved, exact isolation
endpoint and `after_surface` lanes currently emit no receipt; this does not alter their ordinary geometric results.

Each receipt retains the isolation and refined intervals, representative, surface kind, derivative and clearance
margins, guard/resolution, forward material side and complete input binding. `verify_receipt(receipt, expected=...)`
reruns the solver and compares all binding/evidence bytes against independently supplied expected inputs. Verification
is same-runtime reproducibility checking, not portable certification or an independent numerical oracle. Every
receipt has `transport_restart_valid=False`. The separately verified host continuation consumer is described below.
The emitter/verifier are host diagnostics, not a vectorized device transport path or throughput implementation.

Optional regression source is selected with `FLASH_FOIL_RECEIPT_BASELINE_SOURCE`; public tests contain no local path.

## Numerical receipt requirements

The solver must retain the chronological isolation interval and the unique crossing's derivative-sign margin and
other-constraint lower bounds. Preserve the original anchored trajectory (`x`, `u`, momentum, charge and vector
field), solver dtype/evaluator identity, exact selected geometry, unique foil identity and explicit frame binding.
A material-thickness type is not a foil identity. Frame binding must be externally supplied and compared; a receipt
cannot independently certify caller-asserted provenance.

Retain the representative distance and refined bisection bracket separately from the guarded isolation interval.
Adjacent bisection floats do not imply that clearance evaluation has negligible error. In particular, half the final
bracket width is not a certified total root error. Selecting an exact original isolation endpoint also requires
handling that endpoint explicitly, rather than exporting an unrelated final pair of bisection bounds.

Initial receipt support may conservatively decline exact-endpoint, zero-force, continued-start or insufficient-margin
cases while leaving the ordinary geometric result unchanged. Verification must recompute the evidence against the
externally expected binding. Copied immutable evidence prevents accidental aliasing, not malicious forgery or
independent certification of the geometry source. A verifier is not a replacement for a floating-point proof.

### Implemented host-only unchanged-trajectory consumer

`flash_foil_follow.continue_same_trajectory(receipts, expected=..., absolute_s_max=...)` accepts the ordinary
emitter's flat C-order tuple and a mandatory independently owned binding. It recomputes the ordinary evidence once
and compares every supplied receipt to its exact lane. Missing, changed, reordered or unsupported initial receipts
cannot authorize a skip. The requested cap must be finite, nonnegative, exactly representable in the computation
dtype and no greater than the original represented cap. A cap below the consumed isolation upper endpoint refuses.
Mixed Python sequences retain each supplied element's type and exact value before numeric conversion; a boolean
or nonrepresentable integer cannot become a valid float cap through sequence coercion. Explicit ndarrays must
have a real numeric dtype; object, complex, boolean and string arrays are not a numeric cap contract.

The search evaluates the original anchored trajectory at absolute path coordinates, starting at the verified
isolation upper frontier. That frontier is not a new particle position: no path is transported or discarded and no
new exact `after_surface` token is manufactured. Original-cap conditioning, guard and resolution remain in force.
Later crossings, including returns to the same face, remain eligible. If two guarded isolation intervals are
`[L1,U1]` and `[L2,U2]`, the relative interval is `[L2-U1,U2-L1]`, not representative subtraction. These remain
engineering evidence, not outward-rounded floating-point enclosures.

An immutable `FollowResult` separates per-lane `NEXT_HIT`, `CLEAR_TO_CAP`, `UNRESOLVED`, `INVALID_RECEIPT`,
`INVALID_CAP` and `UNSUPPORTED_HIT`. Only the first two are accepted. A geometric next hit can be valid in `result`
while lacking the strict isolation evidence required for `NEXT_HIT`; it is then `UNSUPPORTED_HIT`, never clear.
Callers must inspect `accepted`/`status`, not use `result.valid` as blanket authorization. Next evidence is not a
`GeometricReceipt` and cannot be chained into this consumer. Unsupported intervals contain NaN, not invented bounds.
A cap exactly at the frontier is clear only when verified endpoint clearance also exceeds the retained resolution;
a user-tolerance-dominated frontier can still refuse. Budget exhaustion is unresolved, not no-hit.

No output particle position or direction is produced. `transport_restart_valid` remains false. This host-only,
single-foil diagnostic has no device counterpart and is not a finite-volume transport implementation.
Optional default output/graph regression uses `FLASH_FOIL_FOLLOW_BASELINE_SOURCE`.

Any mutation of the bound trajectory, geometry or frame invalidates reuse. A new direction at an exact surface also
needs a fresh ownership check: reversal can be a legitimate immediate transition, not an already-consumed departure.
No snapping, projection, coordinate/path nudge or arbitrary fuzzy material label is authorized.

## Explicit refusal and primary-byte preservation

The existing `lane_step_emissions(first_exit=...)` hook accepts only a distance. Its caller owns the nonnegative,
resolved answer. It cannot safely consume a `(distance, unresolved)` query by discarding the status or interpreting
an unresolved infinity as certified vacuum. No transactional status API is implemented.

An additional output or observer inside a fused JAX graph can change compiler rounding even when the query reports
no extra hit. Static-infinity tests are insufficient. Existing recorder replay and optimization barriers are tested
implementation measures, not guarantees of executable isolation under arbitrary enclosing programs.

A conservative initial transaction architecture would therefore be host-orchestrated, with these restrictions:

- Invoke an unchanged, separately compiled reference step with its original outputs and retain those buffers privately.
  Bind the byte reference to the exact hook/finite-lane configuration, shape, dtype, backend and compilation context.
  An all-infinity finite-lane hook is not generally equivalent to no hook: it disables the z-only trapped shortcut.
- Validate geometry independently with explicit `CLEAR`, `EXIT`, `UNRESOLVED` and `INVALID` outcomes. A separately
  reconstructed centerline is not automatically the reference's exact hidden MSC path. Acceptance needs conservative
  coverage of both actual legs and relevant recorder replay, a protected trajectory contract, or refusal.
- Initially commit only a certified-clear batch, returning the original reference buffers without recomputation.
  Resolved exits require a separate clipping/certification implementation; they are not clear certificates.
- Refuse the whole batch on any relevant uncertainty. Return the complete input state, including the shared scalar
  RNG iteration, unchanged; expose no speculative photon record. Preserve rollback inputs without buffer donation.
  Per-lane retries require a separate lane-local iteration/RNG contract.

This architecture is a proposal, not an implemented adapter. It does not promise historical bytes across compiler
versions/devices, eliminate conservative false refusals, or meet the existing single-device-loop throughput goal.
It separates a testable byte-preservation contract from geometry coverage and physical closure.

## Discriminating tests before adoption

- Compile once with runtime geometry controls and alternate clear, exit, first-leg unknown and second-leg unknown
  without retracing. Do not let a static no-hit callback stand in for a runtime-dependent query.
- Compare every primary leaf and photon field separately by shape, dtype and bytes. Cover both charges and precisions,
  straight MSC positrons, both field branches, non-MSC, pending loss, trapped and finished lanes, and real brems,
  in-flight annihilation and stopping-pair records including invalid zero slots.
- Keep the reference executable fixed while varying validator complexity and status consumers. Accepted clear buffers
  must remain identical. Exposing new intermediate legs must not be mistaken for proof of the old hidden trajectory.
- Refuse after a prospective emission and verify complete rollback, unchanged shared iteration, zero published births,
  and a resolved retry identical to a clean first attempt.
- Hide a crossing in the second leg; exercise tangencies, corners, endpoint ties, exhausted search and invalid distances.
  Broad-phase uncertainty must refuse, not certify a reconstructed centerline as the reference path.
- Reject altered receipt roots, sides, intervals, trajectory, foil and frame bindings, including equal-thickness foils.
  Approximate representatives of either residual sign never authorize a material restart. Do not assume every runtime
  produces a nonzero residual for a particular synthetic root.

Passing these software tests would not validate random-hinge first passage, stopping/annihilation conventions,
step-size convergence, photon attenuation or showers. Those remain separate reference/physics gates.
