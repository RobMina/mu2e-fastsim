# Combined PE-statistics and travel saved-step control

This separate opt-in arm changes only `PEStatCorrection:true` and `addTravelTime:true` relative to the preserved
deterministic same-CSS baseline. It reuses the existing USER-built analyzer, original saved CSS and actual
conditions/geometry. No C++/RNG substitution, build, transport, baseline rerun, default change or shower-library
population is authorized by this control. Keep all events, photons, correlated entries and zero observations.

## Prospectively fixed hard gates (version 1)

Freeze policy/code/tests before inspecting any combined output. Run a ten-event smoke, then at most the bounded
2,000-event original input. An underpowered smoke is startup/structural evidence, never distribution validation.

1. Pin original-art/baseline/independent-export/propagation-table hashes. Reconcile the full saved baseline against
   independently audited IDs/order/status and all CSS states, then numerically validate that baseline without art
   reexecution. Require exactly the specified event prefix, including empty events.
2. Compare canonical expanded FHiCL against the baseline with exactly the two stochastic switches restored to false.
   Retain process/module labels, fixed-phase unsmeared EWM, Birks/LRU, time cuts and conditions/geometry. Verify
   actual observed ProductIDs rather than inferring branch identity from labels. No separate-arm RNG realization
   identity: stock shares one engine and travel changes the draw positions used by subsequent counts.
3. Require every saved step, actual condition/IOV/CID, event registry and complete ordered summary bit to equal
   the baseline, except event `n_readouts`, which must reconcile with actual rows. Summary members remain exactly
   all accepted steps, even when both sampled readouts are absent. No time/energy/nearest-bin association repairs.
4. Use `calofastsim.poisson.ledger` to predict float32 means and base times for **every accepted CSS times both
   SiPMs**, including zero means and means rounded to zero in the deterministic arm. Exact event/CSSProductID/
   index/SiPM join; positive persisted int32-range counts only, no duplicate/unknown/rejected keys. Zero-initialize
   the complete ledger then fill actual counts, preserving omitted readouts as observations of zero. Require zero
   mean to have zero count. Predictions are source-derived, not directly observed stock intermediates.
5. Check every PE's actual float32 arrival bits against reachable candidates `float32(base_float+table_center)`
   at stock float32 `length-local_z`. Reuse `travel.check_support`, including equal-arrival collapse, CDF plateaus,
   lower_bound boundary reachability and defined signed-int32 depth conversion. Never subtract base time to infer
   the travel bin. Reject nonfinite/undefined-domain input rather than clipping/dropping. Do not demand the
   deterministic/travel-only/Poisson-only NPE or sampled arrival sequence.

Count support plus arrival support is weak: many wrong probability laws pass. No physical-zero, actual-engine,
stock-sampler, stochastic-response, digitization/reconstruction/matching or unbiased-library flag is promoted.
Runtime zero-mean coverage is reported honestly if none occur; synthetic zero-mean fixtures are not stock coverage.

## Three separately scoped prospective ideal-model diagnostic families

The model is **independent exact Poisson counts conditional on fixed input means**, with fresh independent ideal
uniform double travel marks independent of those counts. This is not the actual RandPoissonQ/MixMax law:
CLHEP 2.4.7.1 uses a skew-corrected Gaussian approximation at mean>=100, tabulated/series paths below100, and
variable draw consumption. Realized engine doubles/state are not exported. Pin actual producer, linked CLHEP,
source/header/table provenance; source qualification is not a reproducible-build proof or RNG-law certificate.

A truly IID stream consumed in non-look-ahead stopping-time blocks can implement the stated independence despite
random block lengths. A finite deterministic PRNG is not automatically such a stream. In particular, conditional
travel independence given all counts is a **model premise**, not established by observing count/arrival support.

Each family has fixed alpha .01 and its own empty/power gates. Their **suite union bound is .03**, not .01; no
independence among families is needed for that bound. Require all families to be `not_rejected` before suite wording
`not_rejected`; any rejection wins, otherwise insufficient coverage/power remains explicit. There is no fitted
parameter, new seed, output-driven stratum merge or threshold adjustment to obtain agreement. Numeric CDFs use
float64 SciPy and are not claimed formally error-certified. Statistical rejection remains in the report even when
hard gates pass; it is not automatically a stock defect under this idealized null.

1. **Complete count CDF:** reuse `poisson_stats.diagnostic` unchanged, with every zero-filled accepted input. Four
   strata=positive mean below100/atleast100 times SiPM parity; five input-conditional count thresholds, family20,
   Hoeffding/Bonferroni alpha .01. Zero means stay in the ledger/deterministic gates, not inflated statistical power.
   See `POISSON_CONTROL.md` for the exact existing thresholds and assumptions.
2. **Conditional travel CDF:** reuse `travel_stats.diagnostic` unchanged on all actual PEs, conditional on observed
   counts and the fixed ledger. Stock depth rows times SiPM parity, all time-center thresholds with heterogeneous
   float32 collapse, family `2*nDepth*nTime`, alpha .01, radius plus `delta=2^-24+2^-50` ideal-uniform conversion
   allowance. Every stratum must have total radius<=.05. The diagnostic's historical policy name says
   post-support because it originated in the earlier travel-only analysis; it is frozen **prospectively** here.
3. **Genuinely joint count/mark contrasts:** `combined_stats.diagnostic` uses all accepted inputs, not positive-RO
   selection. Four strata as in count family, four contrasts each, fixed family16 and alpha .01:
   - For each ledger input choose fixed center `j=nTime//2`; threshold `T=float32(base+center[j])`. Let `k` be the
     last center producing the same float arrival and `p=F[depth,k]`. This includes the full collapsed group.
   - Let `N` be zero-filled count and `Y` the number of observed arrivals<=T. Under the ideal marked-Poisson null,
     `Y~Poisson(mu*p_eff)` where `|p_eff-p|<=delta`. Include absent readouts as `N=Y=0`.
   - First contrast is `I=(N>floor(mu)) AND (first generated PE arrival<=T)`, false for N=0. **First means original
     vector generation order, not minimum/earliest arrival or a sorted vector.** Stock fills PETime in generation
     order and the unchanged analyzer copies it without sorting. Expectation at p is
     `(1-PoisCDF(floor(mu),mu))*p`; its rounding-model discrepancy is at most delta.
   - Three thinning contrasts: indicators `Y<=0`, `Y<=floor(mu*p)`, `Y<=floor(mu*p+2*sqrt(mu*p))`, compared to
     heterogeneous ideal Poisson CDFs at each fixed input's `mu*p`, never the CDF of an average mean.
   - Hoeffding radius is `sqrt(log(2*16/.01)/(2*n_stratum))`. Add delta for I, and the stratum average
     `min(1,mu*delta)` for each thinning contrast. Poisson coupling bounds each CDF discrepancy by
     `1-exp(-mu*|p_eff-p|)<=min(1,mu*delta)`. This covers ideal-uniform rounding only, not actual engine law.
   - Empty strata or any **total** radius>.05 are insufficient. All four informative strata are required for
     `not_rejected`. Rejection is an absolute difference strictly greater than its total radius.

Fixed synthetic fixtures include exact-input mutation failures, unknown/duplicate/rejected associations, zero/empty
ledgers, changed counts without cross-arm equality, supported constant samplers, count-dependent first-mark
coupling with correct count and travel marginals, sorting first-mark mutation, heterogeneous collapse, and 1,000
replicates that call the joint implementation under a fixed ideal null. They are not stock RNG certificates.

The no-build driver records frozen policy/checker/all dependent backends, effective config, actual producer and
linked sampler provenance before execution, and refuses overwrites. Logs/outputs are supplied outside the source
tree. Neither process wall time nor synthetic test timing certifies full-chain throughput/device speedup/GPU cost.
