# EX response calibration pilot

The pilot tests whether tracker-tagged crossings, including live channels without a selected pulse, can constrain
CRV response corrections. It is opt-in and does not change `ChainConfig`, the production reconstruction, or any
conditions. The boundary is **fixed tracks and counter geometry to selected per-SiPM pulse charge and time**, not
particle transport or digitization.

## CPU benchmark

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

Each reproducible fixture trains on 9,600 synthetic crossings and holds out another 9,600 on the same eight
channels. The two fixtures inject slopes of opposite signs. `calibration_scores.json` reports absolute errors in
physical units, per-channel correction RMSE, and held-out standardized marked NLL and detection Brier score
(both lower is better). These are separate from the common scorer's paired PE/time shapes, distance-binned
selected-pulse efficiency, and pulse yield. Replay and injected reference share holdout crossings and noise draws;
the shapes and yields share shard IDs. The common efficiency statistic is binomial, not paired. Both fit arms
optimize in float64; the `float32` and `float64` labels refer to prediction and replay only.

| Fixture | Method | Held-out NLL | Brier | Inverse attenuation error (1/mm) | Slowness error (ns/mm) |
|---|---|---:|---:|---:|---:|
| dev | Nominal | 1.342 | 0.245 | 2.40e-4 | 1.50e-3 |
| dev | Positive-only | 1.292 | 0.253 | 4.08e-4 | 4.86e-4 |
| dev | Censored | **0.971** | **0.140** | **1.49e-6** | **8.17e-5** |
| final | Nominal | 1.920 | 0.216 | 1.70e-4 | 1.10e-3 |
| final | Positive-only | 1.991 | 0.244 | 1.34e-4 | 3.10e-4 |
| final | Censored | **1.859** | **0.206** | **1.13e-5** | **4.22e-5** |

The censored arm had no flags and passed the yield check in both cases and both precisions. On an AMD EPYC 9V74
with one BLAS thread, median in-memory fit times over 10 repeats per case were 52 and 58 ms (censored), and 0.73
and 0.78 ms (positive-only). `bench.json` records fit and replay wall times separately, excluding file I/O and
interpreter startup. `stock_rate_per_slot` is null: there is no matched Offline throughput at this boundary.
Both seeds and injected parameters are public and drawn from the fitted model. The `final` fixture is a second
closure check, not an independent or blind validation, and says nothing about detector fidelity by itself.

## Detector-side recipe (keep inputs and outputs private)

Use an extracted, field-off stock EventNtuple covering one run. The reader needs uproot and Awkward and uses the
public EventNtuple schema: [`TrkInfo`](https://github.com/Mu2e/EventNtuple/blob/v06_15_00/inc/TrkInfo.hh) and
[`TrkSegInfo`](https://github.com/Mu2e/EventNtuple/blob/v06_15_00/inc/TrkSegInfo.hh).
It selects a quality KinematicLine fit with a valid in-bounds tracker outer-surface (`TT_Outer`) crossing, ranking
by most active hits, then lowest chi-square per degree of freedom, then input order. The earliest usable crossing
of that track supplies position, momentum direction, and time; signed flight time is extrapolated at the speed of
light. CRV pulses and CRV crossings do not enter track selection. An event without a usable tracker state supplies
no candidate crossings; one with a usable tracker state but no pulses can supply silent crossings.

Select runs recorded without a CRV requirement. Use the same extracted counter geometry and per-channel `CRVCalib`
CSV as the sample; choose the nominal response, pointing margin, and timing/PE cuts on independent control data
before looking at holdout runs. No detector-specific values are assumed here.

```bash
python -m crvfastsim.objective.ex_observations "$NTUPLE" "$COUNTERS" "$CRVCALIB" "$TRAIN_NPZ" \
  --run "$RUN" --margin-mm "$MARGIN" --threshold-pe "$PE_CUT" --time-window-ns "$HALF_WINDOW" \
  --pe-per-mm "$PE_PER_MM" --attenuation-mm "$ATTENUATION" --speed-mm-ns "$FIBRE_SPEED" --delay-ns "$DELAY"

# Export a separate run with the frozen recipe for HOLDOUT_NPZ.
python -m crvfastsim.objective.ex_response "$TRAIN_NPZ" "$CANDIDATE_JSON" \
  --sigma-log-pe "$LOG_PE_WIDTH" --sigma-time-ns "$TIME_WIDTH" --holdout "$HOLDOUT_NPZ"
```

The NPZ has one row per geometrically crossed EX sensor slot, keyed by `(run, subrun, event, channel)`.
The exporter requires entry and exit through the two broad faces with the edge margin on both. `path_mm` is
the chord length, `distance_mm` is the distance from its midpoint to the fibre end, and `track_time_ns` is
the track time at that midpoint. The channel key is `4 * barId + SiPMId`; side 0 (SiPMs 0 and 2) is the
low-coordinate end. `present` is the physical readout mask; `status` bits 0, 1, and 2 make a row ineligible.
Two or more matching sequence-zero pulses passing the frozen PE and time cuts mark an `ambiguous`, ineligible
row. An eligible row with none is `seen=False`, not a zero-valued charge/time measurement. Association is
exact-channel, not the neighbor-bar allowance in V-R4.

The likelihood models preselection log PE and time as conditionally independent Gaussians of fixed width.
For a selected pulse it uses the joint density at the measured values; for an eligible miss it uses
`1 - P(charge passes) * P(time passes)`. Holdout NLL is in standardized coordinates and should only be
compared with the same widths. The analytic gradient fits shared corrections to inverse attenuation length
and inverse propagation speed, plus a common scale and time origin and zero-mean channel gains and offsets.
The positive-only baseline ignores the selection and fits only recorded pulses. Channels without a
selected training pulse, and fits without within-channel distance leverage, are rejected rather than
filled from nominal. The candidate JSON records the training run IDs and input digest and optional
separate-run diagnostics. It is **not** an Offline conditions file. Keep detector NPZs, candidate constants,
and any real reference arms private.

## Gate before using candidate constants

The next step is a private matched-Offline injection study with frozen tracker tags, then replay on independent
EX detector runs. Check pointing and exact-channel mismatches against the V-R4 neighbor-based result; a
misassigned crossing or a missed pulse below Offline's reconstruction turn-on can both masquerade as
attenuation. Anchor the scale, timing, and widths with independent calibrations, test shared slopes against
Offline perturbations, and quantify pulse contamination and model residuals versus distance, layer, and run.
This pilot does not estimate parameter covariance or account for FEB/event-correlated noise. Before expanding
beyond EX, estimate coverage with run/event blocks and propagate correlated calibration variations through
sampled pulses and coincidence finding to event-level veto inefficiency. W5 remains a downstream forward
regression, not a calibration-recovery gate.
