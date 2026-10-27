# Corrected-library shower sampling, donor statistics and source representativeness

## Scope and fixed inputs

This follows `generator-association-repair.md` in the existing layers worktree. The corrected production library,
all 20 strict donors and original baselines stay immutable. Evaluation is the same 500,000 generated events in
shards 11-20, with the retained corrected CCP muon-family response held fixed. Production generated exposure stays
10,000,000 and the ten full-generator shards 1-10 remain the denominator bank. No rate tuning or true-energy
substitution is allowed. This studies the old worktree sampler, not newer main response code.

## Questions kept separate

1. **Sampling:** how far is the retained shower draw from the conditional mean of this exact library?
2. **Finite donors:** how much would that conditional mean change under independent whole-donor-file resampling?
3. **Source:** does a strictly repaired reference-training source (1-10, independent of evaluation 11-20) predict
   materially different shower output under the same evaluated generator population?

## Exact integration for additive observables

For evaluation primary cell `c = (abs(PDG), native KE bin)`, the implemented sampler first accepts with probability
`nshower[c]/ngen[c]`, then chooses a template uniformly. Each template therefore has probability `1/ngen[c]`.
Use the persisted integer `ngen`, including the builder's rounding; no rebinning, clipping or inferred masses.
For a template feature vector X and N evaluation primaries in the cell:

- `m[c] = sum_templates X / ngen[c]`
- `E[total] = sum_cells N[c] m[c]`
- `Cov_sampling(total) = sum_cells N[c] (sum_templates XX^T / ngen[c] - m[c]m[c]^T)`

Features are sector step counts, visible deposited energy and ROOT-flow-aware y/z histograms, with float32 step
positions matching the retained deck diagnostics. These are additive and time-independent. Unique **event/bar**
occupancy and thresholded pulse/cluster response are not additive and are not covered. Cross-sector/bin
covariance is retained. Empty evaluation events remain in exposure and centered event covariance calculations.

The retained sampled output is compared with this expectation using sampling covariance only. Expectation versus
G4 uses the paired event residual `sum_c N_event,c m[c] - G4_event` (or fixed muon plus this mean minus all G4),
with fixed-N covariance of the sum. Sampling variance is not added to this expectation-versus-G4 comparison.
No large-bin covariance inversion or calibrated chi-square claim is planned.

## Donor statistics

Production has 20 equal-exposure files. Resample **20 files per replica**, with a file's multiplicity shared by all
cells and observables; keep the denominator bank fixed. The exact multinomial bootstrap covariance of the mean
is `sum_f (B_f - mean_file_B)(B_f - mean_file_B)^T`, where B_f is that donor file's contribution to the evaluated
expectation. Bootstrap replicas provide quantiles and nonlinear summary diagnostics; synthetic tests check
centering/covariance against physical template duplication. Check every selected population's reach probability
without clipping or silently discarding invalid replicas. Twenty file units limit covariance rank to 19.
This assumes exchangeable independent production files and excludes denominator-bank uncertainty.

The reference source uses ten matched donor/full-generator files and only 500,000 generated events total.
Its source comparison is a diagnostic, not a replacement recommendation or a calibrated systematic. If source
uncertainty is evaluated, resample matched donor/gen file pairs; fixing its own denominator while resampling the
numerator alone would ignore their dependence. Training and evaluation full integer identities must be disjoint.

## Runtime/provenance gates

- Use the existing user-built strict dumper; verify actual plugin loading and unchanged legacy step/particle/
  subrun payloads in every reference-training re-dump. No C++ build or simulation generation is needed.
- Retained shard 1 lacks a standalone dump log. Its original driver's successful return and the neighbouring
  shard's exact naming convention may derive the known input path, explicitly labelled as such. The new exact
  physics-payload comparison is required; do not invent a historical standalone log.
- Compare template/shower projections with the completed strict library, bind full input/source/geometry hashes,
  validate support, and check source/evaluation identity inventories including zero-step events.
- Cross-check compressed template features against actual `_showers_for_gen` emitted rows with identical RNG
  state. Enumerate tiny synthetic outcomes and paired residuals before using analytic moments on data.
- Work one single-threaded job per GPVM, output under a new data directory; never overwrite strict_v2 or baselines.

## Completed study (2026-10-27)

**Decision:** a single shower draw exaggerates the apparent T2/M step deficit; finite production donor statistics
are substantially smaller than that sampling variation. The corrected reference source is not demonstrably
different in the eight sector step/energy totals under the stated file-statistics approximation. This is **not
physics closure**: the conditional nonmuon-energy means remain about 5-7% below G4, T1's position residual and
the unchanged other-bar muon-descendant discrepancy remain, and nonlinear response is outside this study.

### Strict source and exposure certification

Reference training shards 1-10 were re-dumped with the same built isolated plugin: 24,982 accepted events,
25,004 generators, 792,090 particles, 815,699 steps and 3,259 showers. All original physics payloads remain exact.
The 15 PDG mismatches, one negative KE and one same-PDG energy disagreement above 1 MeV are now zero; 16 shower
associations/bins change. The rebuilt reference library has zero unsupported showers. Original generator payloads
in each accepted reference file were compared exactly with its matching full-generator file; pairing is not merely
an equality of event labels. Training/evaluation and production/reference donor identities are disjoint.

A separate full-generator census audit re-dumped **all 20 original full-generator files**, with actual-plugin
loader checks, full-stream role validation, and exact original generator/subrun values and row order. Each has
**50,000 events; all 1,000,000 events have generator rows**. Thus the legacy-inferred event census used here is
complete, including zero-step events. These new files are audit evidence, **not replacements for the denominator
bank**. The exposure normalization and original denominator hashes remain unchanged.

### Sampling versus the mean

The compressed-template calculation matches actual emitted sampler features. Advancing the RNG by the exact
number of descendant-crossing draws reproduces the retained shard-seed shower features exactly, over all ten
held-out shards. Synthetic enumeration verifies the analytic mean/covariance. There are also 256 direct compact
sampler replicas and **10,000 independent multinomial aggregate replicas** of the identical additive-observable
law (not 10,000 full reconstruction runs).

| Sector | Nonmuon step count / G4, retained draw | Production conditional mean / G4 | Reference-source conditional mean / G4 |
|---|---:|---:|---:|
| EX | 0.9368 | 0.9691 | 0.9973 |
| T1 | 0.9380 | 0.9690 | 0.9795 |
| T2 | 0.8535 | 0.9675 | 1.0247 |
| M | 0.7943 | 1.1271 | 1.2331 |

The retained T2 and M **step counts**, respectively, lie below **40/10,000** and **2/10,000** sampled replicas
(one-sided lower fractions 0.004 and 0.0002). Their Monte Carlo 95% binomial intervals are [0.00286, 0.00544] and
[0.0000242, 0.000722]. They are correlated, post-inspection marginal diagnostics—not model p-values or calibrated
multiple-observable tests. In particular, M's low draw is not evidence that this library's *mean step count* is low.
Energy tails are less extreme: T2/M lower-tail fractions are 0.0688/0.0712.

With the retained muon-family contribution fixed:

| Total steps on 500k generated events | Count | Difference from G4 |
|---|---:|---:|
| G4 | 795,615 | -- |
| Retained corrected draw | 776,654 | -2.383% |
| Production conditional expectation | 788,839.04 | -0.852% |
| Corrected reference-source expectation | 796,119.17 | +0.063% |

Production total-step sampling SD is 6,596; donor-file SD of the expectation is 1,418. The paired event-resampling
SD versus G4 is 7,309; combining it with independent production donor covariance gives 7,445, with the remaining
mean residual -6,776. Cross-sector covariance is included in these total errors. The event-resampling estimate is
fixed **event count**, not an exact covariance conditional on each observed generator configuration.

### Nonmuon energy and donor statistics

| Sector | Production mean / G4 | Sampling SD / mean | Production donor SD / mean | Corrected reference-source mean / G4 |
|---|---:|---:|---:|---:|
| EX | 0.9535 | 2.51% | 0.54% | 0.9550 |
| T1 | 0.9468 | 6.43% | 1.41% | 0.9146 |
| T2 | 0.9433 | 4.63% | 0.98% | 0.9803 |
| M | 0.9314 | 8.51% | 1.69% | 0.9989 |

The 2,000 whole-production-file bootstrap replicas keep 20 draws per replica and the same 10M exposure. A bound
certifies every possible resample's reach probabilities are at most one; no invalid replica is clipped/dropped.
The exact multinomial file covariance is used, with rank at most 19. This conditions on the empirical donor
population and assumes independent exchangeable files; it cannot discover an unobserved shower population.

For the source comparison, the source and its full-generator file are paired. First-order file-bootstrap ratio
influences include their numerator/denominator correlation and the **shared-bank covariance** with the production
prediction. All eight source-minus-production step/energy totals are within 1.38 of these estimated SDs. The
production nonmuon-energy residuals versus G4 are approximately -1.86/-0.86/-1.22/-0.89 estimated SDs for
EX/T1/T2/M when paired event, donor and first-order bank terms are included. These modest marginal pulls are not
an equivalence test and do not establish that the 5-7% mean energy residual is physically absent.

The stored species/KE contributions and generator-kinematic quantiles are diagnostics of **retained
shower-producing primaries**, not comparisons of complete generated source populations. They neither prove the
historical production configurations equal nor justify adding true variables to the response model.

### Position residual after integrating shower randomness

The following scores use the expected total-step histogram versus G4, paired event-resampling diagonal variance
plus donor variance, and retained ROOT flow bins. They are **not the same independent-arm deck statistic** as the
single-draw replay. Full covariance is stored, but not inverted or treated as a calibrated chi-square test.

| Sector | Production mean y/z score | Reference-source mean y/z score |
|---|---:|---:|
| EX | 0.90 / 1.67 | 0.84 / 1.66 |
| T1 | 2.78 / 1.19 | 2.07 / 1.01 |
| T2 | 1.48 / 1.06 | 1.13 / 1.03 |
| M | 1.79 / 1.36 | 1.33 / 1.27 |

Flows are zero; no occupied zero-variance/nonzero-residual bin is silently excluded. T1-y remains the clearest
position residual under either source. The muon-family contribution/model is held fixed, so this does not resolve
its known other-bar descendant deficit.

### Support and sparse-source caveats

Four evaluation primaries lie outside the finite denominator bank: photon bin 13 (one), neutron bin 14 (one),
proton bin 15 (two). These are genuine evaluated tail cells, distinct from the repaired misassociation holes.
The sampler supplies zero in them; that is not a physically certified zero response. **None of their events has
any nonmuon G4 step.** The same zero-event-activity bound holds for all evaluated cells with no production shower
template, so those missing cells do not explain this particular held-out step/energy residual. The audit uses
whole-event nonmuon activity as an upper bound, never kinematic truth matching.

In 10,000 matched-generator-file resamples, **5,788 lose support in at least one evaluated cell**. No such replica
is silently removed to form a ratio uncertainty. The source result therefore uses the explicit first-order
influence diagnostic, not purportedly exact nonlinear ratio-bootstrap quantiles. Cells observed to lose support
carry at most 1.63% of a reference-source sector energy mean, or 0.24% of a production sector energy mean in this
sample. This limits their nominal leverage but is not a bound on unknown tail response. More independent,
configuration-compatible generator/source data would be needed to certify tail robustness.

## Evidence, tests and next gate

Root: `/exp/mu2e/data/users/ram2aq/crv_genassoc_20261027/`.

- `source_train_strict_v2/`: immutable plan, ten strict donor completions/comparisons, loader evidence and source
  library. `fullgen_census_v2/certificate.json`: full-event census and exact original payload audit, 20 files.
- `uncertainty_v2/production/`, `reference_source/`: CSR template features, per-file/cell contributions,
  identities, profiles and frozen hashes. `evaluation/`: paired event sufficient statistics and reference-gen checks.
- `production_result/`: analytic sampling covariance, donor covariance, 2,000 donor replicas and reported scalars.
- `source_result_v2/`: corrected-source contrast, matched ratio/bank covariance, exact source/gen-payload checks,
  and denominator-loss audit. `source_result/` is the earlier numerical report without those extra checks.
- `sampling_tails/`: retained-seed feature match and 10,000 aggregate draws; `sampler_check.json`: actual-emission
  projection check. `summary.json`, `support_audit.json`, `details.json`: concise interpretation and census binding.
- `uncertainty_v1/` retains a failed cache-write attempt (`np.savez` argument-name collision); no completed cache
  or physics result was published. The corrected v2 run used new paths and exited zero.

The combined selected suite passes **363 tests** (association/campaign, moments, source/sampling/census checks,
tier-1b and retained denominator/statistics diagnostics); shell syntax and `git diff --check` pass. This is not a
new claim about the full inherited suite.

No transport, simulation generation, baseline rewrite, response tuning, C++ build or main-checkout edit
was made. All study code is in this worktree. The next physics gate is the persistent T1/other-bar
muon-descendant structure and a larger independent held-out validation, rather than replacing the production
shower library on the strength of the old single-seed source swap. Nonlinear cluster response and integration
with newer main require separate validation.
