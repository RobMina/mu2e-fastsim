# EX response calibration pilot

The pilot tests whether track-tagged EX crossings, including live channels without a selected pulse, can constrain
CRV response corrections. It is opt-in and does not change `ChainConfig`, the production reconstruction, the V-R4
preparation, or any conditions. The boundary is **fixed tracks and counter geometry to selected per-SiPM pulse charge
and time**, not particle transport or digitization.

Modules (`crvfastsim.objective`):

- `ex_observations`: EX crossings from the track-CRV association and the matched per-SiPM pulses.
- `ex_response`: one estimator family: the closed-form positive-only fit (`profile_fit`, formerly
  `charge_time_calib`) and the censored likelihood (`fit`), with Hessian uncertainties.
- `ex_synthetic`: public-seed generators for closure tests (the fitted model; Poisson + log-normal mismatches with
  integer or continuous charge).

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
index, so that pulse matching needs no fallback; the probe's time window and PE threshold as stored settings; the
detector-frame origin (or the predicted bar's across-bar coordinate) with the record, so that consumers need no
geometry version; and the probe on all pulses (`probe = 3`) in the ntuple-stage matchers, which run on clustered
pulses only.

```bash
python -m crvfastsim.objective.ex_observations "$NTUPLE" "$COUNTERS" "$CRVCALIB" "$TRAIN_NPZ" \
  --run "$RUN" --margin-mm "$MARGIN" --threshold-pe "$PE_CUT" --time-window-ns "$HALF_WINDOW" \
  --pe-per-mm "$PE_PER_MM" --attenuation-mm "$ATTENUATION" --speed-mm-ns "$FIBRE_SPEED" --delay-ns "$DELAY" \
  --detector-z0-mm "$DETECTOR_Z0"
```

`--detector-z0-mm` is the reco geometry's `mu2e.detectorSystemZ0`, the origin of the association's detector frame:
24171 mm for the v04 nominal tracker, 22935 mm for the commissioning geometry (Offline#2051, which KPP data reco now
uses). The exporter refuses when fewer than half of the crossings lie inside their predicted bar, the signature of a
wrong origin.

**Check on one KPP data ntuple** (CRT run of the v14 association re-reco, commissioning geometry, all pulses kept):
the reader selects one track in about 1 % of the events, the association's probe ran on all pulses (`probe = 3`), and
the chord through the predicted bar agrees with `probe_upred` to 1e-4 mm. About 31 % of the EX sensor rows are
`neighbour` (the probe's pulse on an adjacent bar), so the neighbour rule removes a large share of crossings; its
cause (pointing, or the probe's choice between bars) is an open question for the association. The censored fit
converges on that file's supported channels with a positive-definite Hessian. No fitted constants from detector
data are quoted here.

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
and that probability is `Phi(a)` plus two bivariate-normal orthant probabilities, evaluated exactly (Genz); without
walk it factorizes. These seen/silent terms and their analytic gradient in the five row predictors (mean and log
width of log PE, mean and log width of time, walk) are hep-simlib's experiment-neutral censored crossing likelihood
(`hepsimlib.likelihood.censored`); this module adds the Mu2e parameters (slopes, channel terms, the photostatistics
width) and chains the gradient to them. The gradient matches central finite differences to 1e-6 relative in the
tests, at ordinary points and in the far tails.

Fitted parameters: the shared slopes `alpha` (inverse attenuation length) and `beta` (inverse signal speed), the
per-channel log gains `g` and time offsets `o`, the widths `sigma_0` (`sigma_log_pe`) and `sigma_t`, the
photostatistics factor `excess_noise` (0 = constant charge width, 1 = Poisson counting), and the time walk (ns per
unit log PE). Options fix the widths (`--fix-widths` with `--sigma-log-pe/--sigma-time-ns`; the old behaviour),
fix `excess_noise`, or fix the walk at zero (`--no-walk`). The fit starts from the positive-only closed form, scales
each parameter by the inverse square root of its start curvature, and runs L-BFGS-B in float64, first without walk
and then with it (the walk is boxed so that corr(log PE, time) stays below 0.99). A fit whose largest scaled
gradient exceeds 0.01 sigma is an error; a walk or excess noise held at a box edge is reported `at_bound` and left
out of the covariance.

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

| Generator | Fit | Attenuation pull | Slowness pull | sigma_0 pull | sigma_t pull | walk pull | 1-sigma coverage (att / slow) | Gain RMSE | Fit time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fitted model, constant width | widths fitted | +0.09 +- 1.07 | +0.28 +- 0.90 | -0.38 +- 0.94 | -0.19 +- 0.90 | - | 0.68 / 0.68 | 1.2 % | 9.3 |
| fitted model, PE-dependent width | widths fitted | -0.02 +- 1.00 | +0.21 +- 0.94 | +0.07 +- 0.95 | -0.19 +- 0.86 | - | 0.70 / 0.73 | 1.5 % | 7.9 |
| fitted model, PE-dependent width, walk 0.8 ns | widths and walk fitted | +0.06 +- 1.00 | +0.17 +- 0.86 | +0.03 +- 0.94 | -0.17 +- 0.90 | -0.14 +- 0.82 | 0.72 / 0.78 | 1.5 % | 9.6 |
| fitted model, walk 0.8 ns | walk fixed at 0 | +0.23 +- 1.04 | +1.43 +- 0.86 | -0.02 +- 0.98 | +0.21 +- 0.90 | - | 0.65 / 0.28 | 1.6 % | 7.6 |
| fitted model, constant width | widths fixed at 0.8x true | -4.40 +- 2.12 | -0.19 +- 1.34 | - | - | - | 0.07 / 0.48 | 1.3 % | 3.4 |
| fitted model, constant width | widths fixed at true | -0.03 +- 1.08 | +0.27 +- 0.94 | - | - | - | 0.72 / 0.67 | 1.2 % | 3.1 |
| fitted model, constant width | widths fixed at 1.2x true | +3.30 +- 1.36 | +0.48 +- 0.95 | - | - | - | 0.03 / 0.65 | 1.6 % | 2.8 |
| fitted model, PE-dependent width | constant width fitted | -4.26 +- 2.16 | +0.58 +- 0.98 | +31.28 +- 2.44 | -0.72 +- 0.90 | - | 0.07 / 0.57 | 1.6 % | 5.3 |
| Poisson counts on log-normal yield | widths fixed (0.36, 2 ns) | -5.80 +- 2.44 | +0.17 +- 1.02 | - | - | - | 0.03 / 0.70 | 1.5 % | 3.2 |
| Poisson counts on log-normal yield | constant width fitted | -6.80 +- 2.82 | +0.08 +- 1.02 | -4.15 +- 1.09 | -0.67 +- 0.84 | - | 0.00 / 0.73 | 1.6 % | 5.4 |
| Poisson counts on log-normal yield | widths fitted | -5.83 +- 2.23 | +0.05 +- 1.02 | -2.87 +- 1.27 | -0.63 +- 0.84 | - | 0.00 / 0.73 | 1.5 % | 9.0 |
| Poisson counts, walk 0.8 ns | widths and walk fitted | -5.49 +- 2.23 | +0.10 +- 0.98 | -2.77 +- 1.20 | -0.78 +- 0.92 | -0.84 +- 1.25 | 0.02 / 0.70 | 1.5 % | 10.9 |
| Poisson counts on log-normal yield | widths fitted, threshold 8.5 (continuity) | -1.89 +- 1.31 | -0.00 +- 1.12 | -1.85 +- 0.86 | +1.58 +- 1.08 | - | 0.15 / 0.67 | 1.3 % | 10.1 |
| Poisson + single-PE gain spread (continuous) | widths fixed (0.36, 2 ns) | -4.15 +- 2.11 | +0.01 +- 1.06 | - | - | - | 0.05 / 0.67 | 1.4 % | 3.4 |
| Poisson + single-PE gain spread (continuous) | widths fitted | -2.06 +- 1.32 | -0.04 +- 1.08 | -1.71 +- 0.80 | +1.23 +- 0.98 | - | 0.13 / 0.72 | 1.3 % | 10.2 |
| fitted model, constant width | positive-only closed form | bias -2.6e-04 /mm | +0.28 +- 4.05 | - | - | - | 0.00 / 0.15 | 4.3 % | 0.0 |

- **The fitted model closes.** Pulls are centred with unit spread and the 1-sigma coverage is near 0.68, with
  constant or PE-dependent charge width and with walk; channel gains are recovered to 1.2-1.5 % RMS (positive-only:
  4.3 %, and an attenuation bias of -2.6e-4 /mm, about 25 times the censored error).
- **Width sensitivity.** Widths fixed at 0.8x / 1.2x true bias the attenuation by -4.4 / +3.3 sigma; fitting them
  removes it. A constant width fitted to a PE-dependent truth is biased as much (-4.3 sigma), which is why the charge
  width has the photostatistics term.
- **Walk.** Ignoring a 0.8 ns walk biases the slowness (+1.4 sigma, coverage 0.28); fitting it restores the closure.
- **Model mismatch (Poisson + log-normal) is not a width problem.** With integer counts the attenuation stays biased
  by about -5.8 sigma (-7e-5 /mm) whether the widths are fixed or fitted. Most of it is the count threshold:
  `n >= 9` is `PE >= 8.5` for a continuous model, and the continuity-corrected threshold brings it to -1.9 sigma.
  With a continuous charge (single-PE gain spread 0.15, as a reconstructed PE), fitted widths halve the bias,
  -4.2 to -2.1 sigma (-4.8e-5 to -3.0e-5 /mm). The remaining bias comes from the charge shape, which the log-normal
  surrogate cannot describe: the forward model, not the widths, has to (the planned expected-mode fastsim CRV
  response feeding this likelihood through `nominal_log_pe` / `nominal_time_ns`).
- All 960 fits converge (largest scaled gradient below 0.01 sigma); a mis-specified fit that wants a negative
  `excess_noise` ends with it held at 0 (`at_bound`).
- Fits take 8-11 s for 9,600 crossings on one core with fitted widths (3 s with fixed widths), including the
  Hessian. `predict` in float32 agrees with float64 to 8e-7 in probability, 1.2e-7 in log PE and 1.5e-5 ns.

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

| Fixture | Method | Held-out NLL | Brier | Inverse attenuation error (1/mm) | Slowness error (ns/mm) | Gain RMSE |
|---|---|---:|---:|---:|---:|---:|
| dev | Nominal | 1.249 | 0.245 | 2.40e-4 | 1.50e-3 | 7.0 % |
| dev | Positive-only | 1.341 | 0.304 | 4.08e-4 | 4.86e-4 | 3.5 % |
| dev | Censored | **0.878** | **0.140** | **2.5e-6** (pull +0.1) | **7.8e-5** (pull +0.5) | **1.6 %** |
| final | Nominal | 1.730 | 0.216 | 1.70e-4 | 1.10e-3 | 8.7 % |
| final | Positive-only | 1.921 | 0.275 | 1.34e-4 | 3.10e-4 | 3.8 % |
| final | Censored | **1.669** | **0.206** | **1.23e-5** (pull -1.3) | **4.9e-5** (pull +0.7) | **0.7 %** |

The censored arm passes every histogram and the yield in both fixtures and both precisions (validity 1.0); the
nominal and positive-only arms do not (validity 0 and 0.33 / 0). The held-out NLL now includes the fitted widths
(log PE and ns units), so it is not comparable with the fixed-width numbers of the first version of this pilot.

## Gate before using candidate constants

The fixtures and closures are drawn from known generators; they test the estimator, not detector fidelity. Next: a
private matched-Offline injection study with frozen association tags, then replay on independent EX detector runs.
Check pointing (`neighbour`, `du_line_mm`, the association's `sigv`) and pulse turn-on: a misassigned crossing or a
missed pulse below Offline's reconstruction turn-on can both masquerade as attenuation. Anchor the scale and timing
with independent calibrations, and quantify residuals versus distance, layer and run. The pilot does not model
FEB/event-correlated noise or pulse contamination; estimate coverage with run/event blocks before expanding beyond EX,
and propagate correlated calibration variations through sampled pulses and coincidence finding to event-level veto
inefficiency. W5 remains a downstream forward regression, not a calibration-recovery gate.
