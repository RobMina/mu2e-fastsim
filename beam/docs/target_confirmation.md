# Target-electron confirmation: implementation design, not a certificate

This document is a preparation checklist for opt-in inclusive DIO/internal-RPC
transport to VD13. It is not a frozen preregistration or evidence of equivalence.
The source and same-start transport gates are separate; passing both does not
prove a two-percent bound on independently generated full-chain yields.

## Development checkpoint (2026-11-05)

Work is paused before confirmation. The approved confirmation campaigns have not
run, and source-library admission, upstream independence, positive-event mixing
compatibility and the statistical release gates remain open. Checkpointed code
and synthetic audit tests are not a physics certificate. Source-library empty
outcomes must remain represented: do not fill or drop them to force admission.
The existing empirical support is not silently promoted to an independent-parent
confirmation population when development resumes.

## Frozen scientific scope

The transport denominator is actual emitted electrons, retaining original
multi-electron DIO events. Observe all released primary-path decisions but do
not select the reference on them. A detector-filter counterexample cannot be
removed by zero observed misses in a later sample. Absolute decay/POT rates,
tracker/reconstruction, rare reconstructed high-momentum DIO leakage, internal
RPC pair/positron response and external conversion are outside this scope.

The user-approved development engineering margins are listed below; they are not
a collaboration-approved physics error budget:

| Endpoint | Margin |
|---|---|
| Total reach and p>60 MeV/c with rmax>450/500 mm, fast/G4 | 0.98–1.02 |
| Same yield at rmax>550 mm | 0.95–1.05 |
| Own >450 selection birth-total-energy minus VD13-total-energy median difference | +/-0.02 MeV |
| Difference of central-68% loss half-widths | +/-0.03 MeV |
| Own-selection normalized CDF supremum for p, cos(theta), r, rmax, absolute time and TOF | 0.02 |
| Own-selection loss-tail probabilities above 0.25/0.5/1/2 MeV | absolute difference <= max(0.001, 0.10*q_G4) |
| Coarse joint TV in (p,rmax), (p,cos(theta)), (r,time) | 0.02 |

Joint measures include an explicit outside/not-reached category and all numeric
flow cells, divided by the full emitted-electron denominator. The exact shape
selection family and joint grids still require freezing. Source-only gates are
CDF supremum 0.01 and coarse joint TV 0.02, plus exact spectrum/bin, PDG, weight,
time and multiplicity contracts; daughter/pair variables are included for RPC.
The tail floor does not certify relative accuracy of arbitrarily rare tails.

## Sampling and weighting decisions still required

The planning design has 24 genuinely independent parent arms per process,
two administrative seed blocks per arm, and a separately drawn source-only arm.
Blocks do not automatically introduce a third random effect. Shared upstream
ancestors form one cluster regardless of file labels or seeds. A source-library
construction layout does not establish 24 independent confirmation parents or
silently redefine this planning design; grouping, provenance and weights must
still be frozen. Fresh stopped muons require their own provenance and admission;
they are not yet DIO events.

Declare whether the estimand is the realized empirical library/window mixture
or an expectation over independently generated upstream arms. These are not
interchangeable. Freeze parent inclusion probabilities, event-occurrence
populations, all normalization weights and the handling of random library sizes.
An equal allocation of transport events to parents does not by itself represent
a pooled event-uniform library. Weighting only parents cannot repair a different
within-parent population or a sequential-window mismatch. Repeated wraps retain
exposure, not new independent ancestors. Do not select or refill parents for
favorable stop counts or observed fast/G4 agreement.

DIO event draws must emit every eligible stopped negative muon when the released
generator has selectOne=false. The original electron-counting CLI remains a
different interface. `target_source_events` supplies an explicit-event adapter
with a hash-bound declared empirical occurrence population and separate schedule
and kinematic RNG streams; it does not decide the campaign weights. Independently
drawn fast-source events and released G4 births must use the same declared target
population, with their different event/electron denominators retained.

## Joint statistical representation

For every original event retain paired fast/G4 count contributions, emitted
multiplicity, reach/selection indicators, loss-tail indicators, cumulative
counts and joint-cell memberships. Construct design-weighted parent-level count
vectors, pooling administrative blocks. Ratios are ratios of pooled weighted
counts, not averages of parent-specific conditional probabilities.

The independent-RNG source arm and reused G4 birth census belong in the same
parent vector: shared G4 data and parents induce dependence. A single joint
confidence construction must cover both processes and transport/source families.
A per-endpoint 95% interval is not a family-wide 95% statement.

For independent parents, scatter of the complete parent vectors already includes
within-parent event and seed noise. Do not add an ordinary event bootstrap to
that scatter. Retain event influence diagnostics separately; tiny parent/file
scatter never authorizes zero uncertainty. For a conditional-on-library estimand,
use its appropriate conditional randomization variance instead of silently
claiming upstream-library uncertainty. The statistical construction must match
the declared estimand and sampling mechanism.

## Signed-region projection rules

Given a valid simultaneous region for the underlying signed counts/probabilities:

- Yield ratios require a positive lower bound for the G4 denominator. An
  unidentified or unbounded ratio is inconclusive, not passing.
- Conditional tails include uncertainty in both own-selection denominators and
  q_G4. A sufficient conservative rule is U_absdiff <= max(0.001, 0.10*L_qG4).
- CDF comparisons bound signed F_fast(t)-F_G4(t), not a percentile bootstrap of
  the nonnegative empirical distance. Frozen finite knots need monotonic
  between-knot brackets: F_fast(t_next)-F_G4(t_prev) and the reverse, including
  infinite endpoints. A finite grid evaluated only at its knots is not a bound
  on the continuous supremum.
- For simultaneous signed joint-cell intervals [l_j,u_j] with sum d_j=0, a valid
  TV upper bound is the minimum of half the sum of max(abs(l_j),abs(u_j)),
  sum max(0,u_j), and sum max(0,-l_j). Sparse or empty cells retain uncertainty.
- Invert simultaneous own-arm loss-CDF bands for quantiles, using a frozen
  generalized-inverse definition. For band [L,U], the lower quantile uses U and
  the upper quantile uses L. Bound median and half-width differences from that
  joint region; insufficient support or grid resolution is inconclusive.

Pass requires the entire simultaneous region inside every specified margin.
Fail requires evidence that the true endpoint is outside a margin; all other
cases are inconclusive. No non-rejection-as-equivalence, sequential expansion,
outcome-dependent bins, tolerance widening or candidate tuning on confirmation.

## Calibration is not yet settled

A parent-only joint multiplier/max-t construction can exploit the signed
covariance and pairing, but with only 24 parents its coverage is approximate.
It requires explicit heterogeneity/sparsity calibration; a zero estimated
variance is not a valid zero-width interval. Resampling both parent vectors and
all constituent events naively double-counts lower-level noise.

A distribution-free finite-sample alternative uses simultaneous bounded-mean
concentration slabs with a preallocated global error budget. It needs known
multiplicity/weight bounds and a matching independent-parent sampling design.
Its range remainder remains nonzero with zero scatter and can be too wide for
these margins at 24 parents. Existing pilot maxima are not known population
bounds. No tight distribution-free confidence or adequate power is established
by the approved event counts alone. The algorithm choice, global error
allocation, quantile grids and parent model remain prelaunch blockers.

## Release checklist

Before any positive confirmation: finish parent admission/ancestry; freeze the
estimand, weights, windows, candidate hashes, seeds, bins and statistical code;
validate interval coverage on synthetic sparse, correlated and heterogeneous
scenarios; evaluate power using existing pilots without retuning physics; and
verify all-event generation/read/parent multiplicity contracts. If adequate
precision needs a different design or cap, obtain a decision rather than
expanding automatically. Development inventory distances and illustrative
critical values remain diagnostic and cannot stand in for this procedure.
