# Travel-only saved-step control

This opt-in control runs stock `CaloShowerROMaker` on the original saved compressed steps, with travel enabled and
PE statistics disabled. It changes no C++ source, library, default or shower model, and reuses the saved no-travel
baseline rather than rerunning it. It is not a digitization/reconstruction or unbiased-library control.

## Prospective hard gates (version 1)

These checks are fixed before inspecting travel-only output:

1. Exactly the independently specified input-event prefix, including empty events. The baseline is itself reconciled
   against the independently audited numeric export (event IDs/order/status and every saved CSS state), then passes
   the existing deterministic checker. Neither reconstructed steps nor matched-by-value rows replace saved input.
2. Canonical expanded FHiCL differs from the saved baseline **only** by `addTravelTime: true`. The existing process
   and module labels are kept, so branch identities are comparable; any observed registry mismatch fails rather
   than assuming ProductID stability from labels alone. Fixed-phase unsmeared stock marker, PE statistics disabled,
   Birks/LRU, conditions, geometry and start/end/buffer remain as in the baseline.
3. Event and saved-step tables (including actual conditions, IOVs/CIDs) and the complete ordered summary tables
   are unchanged. Readouts join by exact event + CSS ProductID/index + SiPM, not tied sort order, time or energy.
   Require the same complete readout-key set and each readout's exact NPE. Accepted zero-PE steps stay in summaries;
   no inference of physical zeros from empty products is allowed.
4. Every observed PE-time **bit pattern** belongs to its conditional reachable set
   `float32(baseline_stored_float_time + table_float_center)` at stock float32 depth `length - local_z`.
   Do not subtract a folded time to infer a sampled bin. Distinct centers that collapse to one stored float are
   grouped by both lower and upper binary searches before testing reachability. Interior equal-CDF bins are
   unreachable under `lower_bound`, but a collapsed group is supported if any member bin is reachable. The first
   bin remains supported at uniform zero. Retain all finite late source times and negative/high depth clamps in the
   stock-defined int-conversion domain; reject invalid/nonfinite data, never clip/drop to obtain agreement.

A ten-event smoke precedes the full bounded saved input (at most 2,000 events). The smoke is only startup/structural
coverage, not a basis for runtime extrapolation. One CPU payload at a time, no new Geant4, no baseline rerun.
`scripts/check_travel_control.py` reports exact-invariant failures or nonzero unsupported-arrival counts; numeric
support failures retain their report. Inputs/baseline/reference/table and effective configuration/source hashes are
recorded; output paths refuse overwrite. `scripts/travel_control.sh` verifies the existing user-built analyzer,
freshness, backing/envset, baseline/input hashes, exact expanded-config delta and unchanged resolved condition hashes
before running. Logs/outputs are supplied by the operator outside the source tree. It never builds or transports.

## What passing means

Passing establishes the tested same-CSS count/association/summary invariants and **arrival support**, not the
probabilities with which that support is sampled. A degenerate sampler always returning one supported center could
pass these hard gates. No CLHEP/JAX same-seed equivalence, production RNG, distributional response or new-device
exactness is certified. `POLICY` in the checker makes these false flags explicit.

The existing public-stock propagation control validates the exercised cache-building and inverse-CDF behavior with
prescribed draws; it does not supply the production travel realizations for this arm. Statistical agreement needs
another prospectively frozen test, not a post-hoc interpretation of support success:

- Use the actual CDF and the collapsed stored-arrival groups. If `G(x)=P(float32(engine_double)<=x)`, their bin
  masses are `G(F[0])` and `G(F[j])-G(F[j-1])`, aggregated over equal arrival values. Continuous CDF differences
  alone omit float-rounding and the actual engine's discrete law.
- Pin engine type/version and choose its exact law or an explicitly approximate model with a justified fixed
  discretization allowance. Predeclare strata, statistic, family-wise error allocation and power/sample-size gates.
  Calibrate heterogeneous/sparse categorical null coverage conditionally on the fixed input ledger before testing.
- Preserve all events/zero means, and separate travel-only, Poisson-only and combined arms. No fitted parameter,
  seed searching, response reweighting or event removal is an acceptance method.

Until an actual-engine test passes, `stochastic_distribution_validated` and `stochastic_response_certified` remain
false even if every hard gate above passes. No Poisson/combined arm is launched by this control.

## Separately frozen post-support ideal-model diagnostic

After obtaining only the support/invariant results, and **before inspecting the arrival histograms or any CDF
contrasts**, the optional `--ideal-uniform-diagnostic` is fixed as follows. This is new analysis of already-recorded
travel output, not an original blind full-chain validation or actual MixMax-law test.

- Eighty strata: 40 stock depth rows times two SiPM parities. Fixed 47 conditional thresholds per stratum; the
  correction family includes all 3,760 pairs even for empty strata. No histogram-driven bins/strata or merging.
- For each readout, threshold `j` is its float32 base-plus-center arrival. Let `k(j)` be the last table bin with the
  same stored arrival as threshold `j`. The conditional ideal CDF is `F[depth,k(j)]`; sum this weighted by each
  readout's NPE. An arrival indicator uses its first equal candidate index `<=j`. This handles heterogeneous
  float32 collapse before aggregation rather than naively pooling bin counts.
- Under independent ideal uniform doubles conditional on the fixed input/count ledger, each indicator is an
  independent (not necessarily identical) Bernoulli. Two-sided Hoeffding plus Bonferroni yields radius
  `sqrt(log(2*3760/0.01)/(2*N_stratum)) + (2^-24 + 2^-50)` with at least 99% simultaneous model coverage.
  Thresholds need not be independent of one another.
- The additive allowance applies **only** to the ideal-double model: round-to-nearest float32 changes
  `P(float32(U)<=F)` from `F` by no more than half the largest adjacent float gap on [0,1], bounded conservatively
  by `2^-24`; `2^-50` additionally covers choosing a nearest-representable uniform double rather than a real U.
  CDF values themselves are already stored float32 inputs, not approximate fitted probabilities. This argument
  makes no statement about the marginal law or serial dependence of the actual deterministic MixMax stream.
- Report rejected pairs, counts, per-stratum max deviations and radii. Empty strata or radii above .05 are
  explicitly insufficient coverage/power; no distribution PASS is emitted unless every stratum is informative,
  and even then the wording is only ideal-model `not_rejected`, never production RNG validation.
- Fixed synthetic controls include first-bin/last-bin/midpoint samplers, uniform-bin rather than inverse-CDF,
  heterogeneous collapse and a 1,000-replicate ideal multinomial null coverage fixture. No seed matching.

`calofastsim.travel_stats.POLICY` stores this diagnostic separately from the original hard-gate policy. The optional
statistic does not alter the previously fixed support gate or classify stochastic response as certified. The source
engine review pins MixMaxRng/CLHEP 2.4.7.1 (from expanded service default and stock createEngine); production engine
state and individual double draws were not exported. No new art run is needed for this diagnostic.
