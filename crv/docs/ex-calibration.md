# EX response calibration pilot

The pilot tests whether track-tagged EX crossings, including live channels without a selected pulse, can constrain
CRV response corrections. It is opt-in and does not change `ChainConfig`, the production reconstruction, the V-R4
preparation, or any conditions. The boundary is **fixed tracks and counter geometry to selected per-SiPM pulse charge
and time**, not particle transport or digitization.

Modules (`crvfastsim.objective`):

- `ex_observations`: EX crossings from the track-CRV association and the matched per-SiPM pulses.
- `ex_response`: one estimator family: the closed-form positive-only fit (`profile_fit`, formerly
  `charge_time_calib`) and the censored likelihood (`fit`), with Hessian uncertainties.
- `ex_synthetic`: public-seed generators for closure tests (the fitted model, and a Poisson + log-normal mismatch).

## Crossings from the track-CRV association

Tracks and crossings come from the collaboration's track-CRV association (Offline `TrkCrvMatch`; EventNtuple
`<trk>crv`, `<trk>crvxings`, EventNtuple >= v06_17_00), not from a private extrapolation:

- One quality KinematicLine track per event (status, good fit, active hits, fit consistency, chi2/ndof), ranked by
  most active hits, then lowest chi2/ndof, then input order. CRV data do not enter the track choice.
- Its association record must exist (`matched`) with the layer probe run (`probe >= 1`). Each in-bounds EX crossing
  gives, per layer, the probe's predicted bar (`probe_bar`), position along the bar (`probe_upred`) and track time
  at the layer (`probe_tpred`). The fibre distance to each end is `half length +- probe_upred`; side 0 (SiPMs 0 and
  2) is the low-coordinate end.
- The only local geometry is the straight chord through the predicted bar, from the association's crossing position
  and direction: its length is `path_mm`, and it must enter and leave through the two broad faces inside the edge
  margin. `du_line_mm` records the chord's along-bar position minus `probe_upred` (zero for a field-off line).
- Crossings whose probe pulse lies on a neighbouring bar (`probe_dbar` not 0 or 127) are flagged `neighbour` and are
  ineligible by default (`--keep-neighbour` keeps them): they are pointing mismatches, which would otherwise look
  like missed pulses.

**Fallback, per-SiPM pulses.** The association stores no per-SiPM pulse: its probe sums the selected bar's
in-window pulses over both ends (`probe_pe`) and keeps one time difference. The calibration needs per-SiPM charge
and time, so the exporter matches `crvpulses` on the association's predicted bar: exact channel `4 * barId +
SiPMId`, sequence 0, PE and time inside the frozen cuts. This step is labelled in the output (`pulse_match =
"crvpulses_exact_sipm"`). Two such pulses make the row `ambiguous` (ineligible); none makes it `seen=False`, not a
zero-charge measurement.

**Unclustered pulses are required.** With `crv.keepUnclusteredPulses: false` (the EventNtuple default) `crvpulses`
holds coincidence members only, and a missed pulse would include every pulse outside a coincidence. The exporter
refuses such an input unless `--allow-clustered-only` is given.

Follow-up items for the association (not changed here): per-SiPM (per-end) probe charge and time with the pulse
index, so that pulse matching needs no fallback; the probe's time window and PE threshold as stored settings; and
the probe on all pulses (`probe = 3`) in the ntuple-stage matchers, which run on clustered pulses only.

```bash
python -m crvfastsim.objective.ex_observations "$NTUPLE" "$COUNTERS" "$CRVCALIB" "$TRAIN_NPZ" \
  --run "$RUN" --margin-mm "$MARGIN" --threshold-pe "$PE_CUT" --time-window-ns "$HALF_WINDOW" \
  --pe-per-mm "$PE_PER_MM" --attenuation-mm "$ATTENUATION" --speed-mm-ns "$FIBRE_SPEED" --delay-ns "$DELAY"
```

Use an extracted, field-off EventNtuple of one run with the association branches and unclustered pulses, recorded
without a CRV requirement, with the same counters file and per-channel `CRVCalib` CSV as the sample. `present` is
the physical readout mask; `status` bits 0, 1 and 2 make a row ineligible. Choose the nominal response, margin and
cuts on independent control data before looking at holdout runs. Keep detector NPZs and candidate constants private.

## Likelihood

For a crossing at fibre distance `x` on channel `c`, relative to the expected log PE `q0` and time `t0` supplied per
crossing (`nominal_log_pe`, `nominal_time_ns`):

```
log PE    ~ Normal(mu_q, sigma_q),   mu_q = q0 + g[c] - alpha x,   sigma_q^2 = sigma_0^2 + excess_noise / exp(mu_q)
time | PE ~ Normal(mu_t - walk (log PE - log_pe_ref), sigma_t),     mu_t = t0 + o[c] + beta x
```

A selected pulse contributes the joint density at its log PE and time; an eligible crossing without one contributes
the probability of failing `PE >= threshold` or `|time - t0| <= window`. With walk, log PE and time are correlated
and that probability is `Phi(a) + Phi2(-a, kL; r) + Phi2(-a, -kU; -r)`, evaluated exactly with the bivariate normal
CDF (Plackett/Genz rule, `|r| <= 0.95`); without walk it factorizes. The analytic gradient matches central finite
differences to 1e-6 relative in the tests, at ordinary points and in the far tails.

Fitted parameters: the shared slopes `alpha` (inverse attenuation length) and `beta` (inverse signal speed), the
per-channel log gains `g` and time offsets `o`, the widths `sigma_0` (`sigma_log_pe`) and `sigma_t`, the
photostatistics factor `excess_noise` (0 = constant charge width, 1 = Poisson counting), and the time walk (ns per
unit log PE). Options fix the widths (`--fix-widths` with `--sigma-log-pe/--sigma-time-ns`; the old behaviour),
fix `excess_noise`, or fix the walk at zero (`--no-walk`). The fit starts from the positive-only closed form, scales
each parameter by the inverse square root of its start curvature, and runs L-BFGS-B in float64.

**Uncertainties.** The covariance is the inverse of the observed information: the Hessian of the summed NLL at the
optimum, assembled from per-crossing Hessians in the row predictors (central differences of the analytic gradient)
and the block structure of the parameters. It is reported in the gauge of the output: `log_scale` and `time_zero_ns`
are the channel means, `log_gain` and `time_offset_ns` sum to zero (their covariance rows sum to zero). The JSON holds
the errors of every reported parameter and the covariance of the shared terms; `--covariance-out` writes the full
gauge-fixed matrix. A fitted `excess_noise` at its bound (0) is reported as `at_bound` and is not in the covariance.

**Forward model interface.** The fit only sees `nominal_log_pe` and `nominal_time_ns` per crossing. The analytic
nominal of the exporter can be replaced by any forward model's expected log PE and time per crossing (for example a
future expected-mode fastsim CRV response); the fitted corrections are then relative to that model.

**Positive-only closed form.** `fit(..., include_missing=False)` (CLI `--positive-only`) and `profile_fit` solve the
weighted least squares of the recorded pulses with the channel intercepts profiled out, optionally with per-pulse
inverse-variance weights (`charge_weight`, `time_weight`). It ignores the selection and is the start point and the
uncensored baseline; its slope errors come from the residual scatter.

## Closure, pulls and model mismatch

`crv/scripts/ex_calibration_closure.py` (60 toys per case, 8 channels x 1,200 events = 9,600 crossings, threshold
9 PE, window +-4 ns, true `sigma_t` 2 ns). Pull = (fit - truth) / Hessian error; mean +- sd over the toys.

CLOSURE_TABLE

Earlier validation of the fixed-width fit (2026-11-19, same toys): attenuation pulls -3.5 +- 2.0 and +2.6 +- 1.4 for
fit widths 0.8x and 1.2x true, and -4.1 +- 3.0 under the Poisson + log-normal mismatch.

## CPU benchmark fixtures

From the repository root with NumPy and SciPy installed (no JAX, ROOT, Offline, or GPU required):

```bash
export PYTHONPATH="$PWD:$PWD/core:$PWD/crv"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
work="$(mktemp -d)"
for phase in dev final; do
  out="$work/$phase"
  python -m benchmarks.fastsim_score.crv_calibration "$out" --phase "$phase"
  for arm in nominal positive_only censored; do
    python -m benchmarks.fastsim_score.score "$out/submissions/$arm" "$out/reference" \
      --out "$out/$arm-score.json"
  done
done
```

Each fixture trains on 9,600 synthetic crossings drawn from the fitted model and holds out another 9,600 on the same
eight channels; the two fixtures inject slopes of opposite signs. `calibration_scores.json` reports absolute errors,
channel RMSE, held-out NLL per crossing and detection Brier score, and, for the censored arm, the Hessian errors and
pulls. The replay draws the held-out crossings from the same standardized noise at each arm's widths, in float32 and
float64, into the common scorer format. The censored arm fits widths and walk.

FIXTURE_TABLE

## Gate before using candidate constants

The fixtures and closures are drawn from known generators; they test the estimator, not detector fidelity. Next: a
private matched-Offline injection study with frozen association tags, then replay on independent EX detector runs.
Check pointing (`neighbour`, `du_line_mm`, the association's `sigv`) and pulse turn-on: a misassigned crossing or a
missed pulse below Offline's reconstruction turn-on can both masquerade as attenuation. Anchor the scale and timing
with independent calibrations, and quantify residuals versus distance, layer and run. The pilot does not model
FEB/event-correlated noise or pulse contamination; estimate coverage with run/event blocks before expanding beyond EX,
and propagate correlated calibration variations through sampled pulses and coincidence finding to event-level veto
inefficiency. W5 remains a downstream forward regression, not a calibration-recovery gate.
