# PE-statistics-only saved-step control

This opt-in control changes only `PEStatCorrection: true` in the saved deterministic response configuration.
`addTravelTime` stays false. The same original saved CSS, stock fixed-phase unsmeared event marker, actual
conditions, Birks/LRU, geometry and process/module labels are retained. No transport, baseline rerun, C++ change,
build, RNG substitution, default change or library population is part of this control.

## Prospectively fixed hard gates (version 1)

Before inspecting any PE-statistics output, fix these checks and the diagnostic below:

1. Pin the original input, saved baseline and independent numeric export. Reconcile the full baseline against
   independently audited event IDs/order/status and CSS states, then run the existing deterministic checker on
   the baseline numerically, never rerun baseline art. Select exactly the specified event prefix (at most 2,000).
2. Require canonical expanded FHiCL to differ only in `PEStatCorrection`, with travel false. The driver checks
   unchanged resolved condition hashes, existing USER-built analyzer freshness/hash, pinned backing/envset and
   clean linkage before stock execution. Actual observed ProductIDs, not just module labels, must agree.
3. Every field/bit of event, saved-step and ordered summary tables agrees with the baseline except event
   `n_readouts`, which is independently reconciled with the actual readout rows. Accepted steps, ordered
   CSS-to-summary membership and summary energy/time are unchanged even when both sampled counts are zero.
4. Build a complete ledger of **every accepted CSS times both SiPMs**, including zero means and means rounded to
   zero in the deterministic arm. Predict mean from actual float32 corrected energy and actual per-SiPM yield;
   these remain source-derived predictions, not observed stock intermediates. Join by exact event + CSS
   ProductID/index + SiPM. Reject duplicate, rejected-step or unknown readouts. Each persisted readout has a
   positive integer count in int32 range. Initialize the entire ledger to zero, then fill persisted counts:
   an omitted readout is a zero observation, never a missing statistical sample or reason to drop the input.
5. Require zero mean to give zero count, and every observed PE time to have precisely the float32 bits of its
   predicted folded no-travel time. No nearest-time matching, clipping, tail removal or finite-probability count
   cutoff. Finite nonnegative means at most 2e9 define this bounded input domain; an out-of-domain input stops
   the control rather than being silently removed. All empty events, photons and correlated entries are kept.

Passing these hard gates establishes input/condition/summary/identity consistency and weak count support only.
It does **not** establish that counts have the right probability law. Even deterministic rounding can pass.
Count zero for an accepted response is not proof of a physical-zero incident shower.

## Stock sampler source qualification

Stock calls `CLHEP::RandPoissonQ::fire(double(mean_float))` on the producer's engine. In CLHEP 2.4.7.1 the actual
`shoot(engine, mean)` branch is `mean < 100` versus `mean >= 100` (not the inclusive boundary suggested by a header
comment):

- Nonpositive mean returns zero without drawing. Positive mean below 10 uses a direct CDF series. Mean from 10
  through below 100 uses tabulated base means and a residual-mean draw, sometimes reusing a subinterval of the
  first uniform and sometimes drawing a second uniform. Floating CDF boundaries/rare-tail termination matter.
- Mean at least 100 uses `RandGaussQ` and a skew-corrected quadratic Gaussian approximation, clamps negative
  results to zero and values above 2e9 to 2e9, then truncates to integer. It is **not exact Poisson**.
- `RandGaussQ` uses tabulated/interpolated Gaussian quantiles and special tail evaluation. The expanded stock
  engine default is MixMaxRng. Neither the realized doubles nor engine state are exported in this analyzer.

Pin source/header/table and loaded library provenance. Do not infer the actual discrete/serial law from comments,
ideal continuous uniforms, matching seeds, or NumPy/JAX's Poisson implementation. Combined travel plus statistics
shares the engine and changes draw consumption; that separately executed arm is documented in
`COMBINED_CONTROL.md`, with the same production-RNG limitations.

## Separate prospective ideal-Poisson CDF diagnostic

This is deliberately **not** a stock-RandPoissonQ-law acceptance test. A rejection is meaningful disagreement with
an idealized model, not automatically a stock defect; non-rejection does not establish engine independence,
algorithm exactness or production stochastic response.

- Condition on the complete fixed mean/SiPM ledger, not on which counts were positive. Zero means remain in the
  ledger and are deterministically checked; their ideal CDF is identically one, so they do not inflate test power.
- Four fixed strata: positive means below 100 versus at least 100, times two SiPM parities. No data-driven strata,
  merging, seed searching, fitted response or mean substitution.
- Five integer thresholds for each input: 0, floor(max(0, mu-2 sqrt(mu))), floor(mu), floor(mu+2 sqrt(mu)), and
  floor(mu+4 sqrt(mu)). At each threshold compare the sum of observed count indicators to the sum of conditional
  ideal-Poisson CDFs, evaluated with float64 `scipy.special.pdtr`. Thresholds vary with the fixed input mean;
  heterogeneity is retained before summation, not replaced by an average mean. Numerical CDF evaluation is not
  claimed as a formally certified error bound or a derivation of the CLHEP approximation law.
- Under conditional independent exact Poisson counts, the indicators are independent nonidentical Bernoulli.
  Two-sided Hoeffding plus Bonferroni gives radius `sqrt(log(2*20/0.01)/(2*N_stratum))` for nominal 99% simultaneous
  ideal-model coverage. The family is always 20, including empty or redundant pairs. Independence among
  thresholds is not needed. This bound describes the mathematical ideal CDF, with the stated numerical caveat.
- All four strata must be nonempty with radius at most .05 for the wording `not_rejected`. Empty/underpowered
  strata give `insufficient_coverage_or_power`; any contrast beyond its fixed radius gives `rejected`. Retain
  numeric reports in all cases; the diagnostic does not overwrite the hard-gate decision.
- Fixed fixtures test deterministic rounding, all-zero sampling, excess dispersion, biased counts, heterogeneous
  CDFs, omitted positive-mean zeros, zero means, empty/underpowered cases, and a 1,000-replicate ideal null check.
  Synthetic tests are not stock executions or production engine certification.

A ten-event existing-art smoke precedes the full bounded arm. Preserve all reports/output and reuse the baseline.
Record source/config/input/output hashes before/after execution; refuse overwrites. No speedup/device/GPU-cost
claim follows from process timing. Production RNG, stock sampler law, stochastic response, digitization,
reconstruction, matching and unbiased-library certification flags remain false regardless of this diagnostic.
