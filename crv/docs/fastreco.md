# Fast CRV reconstruction (stage 1): goals, validation, prototype

Status: **goals and thresholds agreed 2026-10-16.** First comparisons on existing digis done 2026-10-16 (Results).

## Purpose

A response or alignment point of the CRV MC/data loop re-runs the chain from cached `CrvStep`s: response, waveforms,
digis, pulse reco, coincidences, observables. The fast simulation (`plan.md`) removes Geant4 and moves the response to
GPU; this part replaces the CPU digitisation and reco after it, so one point runs without art. The stock Offline chain
stays the validation reference (path A in `plan.md` section 6).

## Two paths, compared before choosing

| path | chain | why |
|---|---|---|
| waveform | SiPM charges -> waveforms (port of `CrvWaveformsGenerator` + `MakeCrvWaveforms` + `MakeCrvDigis`) -> pulses (port of `MakeCrvRecoPulses` + `CrvRecoPulsesFinder`) | thresholds, zero suppression, double pulses and failed fits are modelled where they happen; the response knobs `minVoltage`, noise and FEB jitter act at this level |
| direct | SiPM charges -> pulses through a response model learnt from stock (charges, pulses) pairs | cheaper; a candidate low-fidelity tier |

They are compared on the same V-R2/V-R4 observables and on cost per event; the waveform path is the reference where
they disagree.

## Goals

| ID | Goal | Target |
|---|---|---|
| R1 | pulse reco + coincidences, one A100 | >= 100x one CPU core running `CrvRecoPulsesFinder` + `CrvCoincidenceFinder` (>= 20,000 accepted events/s/GPU) |
| R2 | waveform generation, one A100 | >= 50x one CPU core |
| R3 | one point end to end (response + waveforms + reco + objective) on 1 M cached accepted events | <= 15 min on one Polaris node (stretch 3 min) |
| R4 | container | the fast-sim image + < 0.5 GB; no CVMFS; cold start <= 5 min |
| R5 | reproducibility | bit-identical per seed and backend; CPU/GPU statistically equal |
| R6 | CPU fallback | runs on CPU for tests and small samples at >= 5x the art chain per core |

## Validation

Matching by truth particle where truth is used, all sectors, no free scale factors.

| level | test | threshold |
|---|---|---|
| V-R1 waveforms | same SiPM charges into the fast and stock waveform/digi code, noise off then on | sample-identical without noise; with noise, per-channel ADC distributions KS p > 0.01 |
| V-R2 pulses | stock digis -> fast reco vs stock reco, per pulse (event, channel, sequence index) | identical pulse finding; failed-fit flag agrees on >= 99.5 %; PE within 0.5 % and time within 0.1 ns for >= 99 % of good fits; no sector-dependent bias |
| V-R3 coincidences | stock pulses -> fast coincidence finder vs stock | identical cluster lists |
| V-R4 end to end | stock chain vs fast path from the same `CrvStep`s, per sector | response observables: binned ratio within +-3 %; coincidences per event at the data operating point within 1 %; EX layer efficiency within 0.3 points; PE median within 2 % |
| V-R5 objective | the loop objective from both paths on a grid of knob points | same minimum within the loop noise; difference <= 1/10 of its spread over the scanned range |
| guard rail | | every fast-minus-stock difference <= 1/10 of the present MC-data difference |

## Reference data and code

- `art/CrvFastSim/src/CrvRecoTableDumper_module.cc` and `art/dump_crvreco_ref.sh mc|data`: stock chain tables (SiPM
  charges, digi MC voltages, digis, pulses, clusters, conditions) from an extracted dts file (MC) or a KPP raw file
  (data). The KPP data has the real noise, double pulses and failed fits.
- `crvfastsim/reco/`: `python3 -m crvfastsim reco-vr2|reco-vr2jax|reco-bench|reco-vr1|reco-direct|reco-cost REF.root ...`
  (README section 10).
- Stock cost for scale (TimeTracker means per event): `CrvRecoPulsesFinder` 1.6 ms (extracted MC) and 3.7 ms (KPP
  data), 97 % of it in `TGraph::Fit`; the CRV response 19.8 ms per accepted event (`plan.md` section 3).

## Results

| check | sample | result |
|---|---|---|
| unit tests | synthetic waveforms | fit recovers injected pulses; peak finder, pedestal rule, digi grouping, waveform linearity, ZS |
| stock oracle (`vr2stock`) | MC 5,000 ev / data 5,000 ev | MC: 161,912 pulses identical (PE exact, time < 1e-9 ns): the dump, the digi grouping and the conditions reproduce the art job. Data: 149,210 vs 149,211; the missing pulse is the stock fit's state carried between waveforms (below) |
| V-R2 (`vr2`) | MC, 22,030 events with CRV digis, 709,883 pulses | **pass**: finding identical; failed-fit flag agrees 99.992 %; 99.97 % of good fits within 0.5 % PE and 0.1 ns (PE p99 3.7e-5, time p99 6e-4 ns) |
| V-R2 (`vr2`) | KPP data 124155_000001, 45,072 events, 1,313,844 pulses | every quantitative criterion passes (flag agreement 99.973 %, 99.966 % of good fits within PE/time); "finding identical" fails on 11 waveforms out of ~0.9 M (7 fast-only, 6 stock-only pulses), explained below |
| V-R2 after the fit polish (Gauss-Newton steps after LM, 2026-10-16) | MC / data | MC still passes, flag agreement 99.996 %, 99.974 % of good fits within PE/time; data flag agreement 99.979 %, 99.967 % within, same 11 waveforms. CPU time per event on a busy gpvm04: 3.4 ms (MC), 7.4 ms (data) |
| V-R1 (`vr1`) | MC, both arms noise-free (reference `crvreco_mcnonoise`, `CrvWaveforms.noise: 0`), 22,095 events | **every stock digi reproduced sample-identically** (565,579 of 565,579, max ADC difference 0). The fast path makes 35 more (0.006 %), all on FEBs with no stock digi in that event, whose per-FEB TDC0 the reference cannot supply (the test then uses TDC0 = 0); not a disagreement. TDC0 is drawn per FEB and event (spread within (event, FEB) = 0; within (event, ROC) up to 12.47 ns) |
| direct path v0 | MC, odd events (290,244 charge clusters) | first-pulse PE median +0.05 % (KS p 0.64), time median 0.7 ns (KS p 1.0), beta median +0.13 % but KS p 5e-7; found fraction 88.50 % vs 88.54 %, but wrong at threshold (6-8 PE: 25 % vs 14 %; 8-10 PE: 49 % vs 66 %): the charge bins were too coarse there |
| direct path v1 | same split | 0.5 PE charge bins to 20 PE, binned in prompt fraction (charge in the first 15 ns) instead of time spread. Threshold fixed: 6-8 PE 13.8 % vs 14.2 %, 8-10 PE 63.5 % vs 66.0 %, 4-6 PE 0.4 % vs 0 %. PE median +0.13 % (KS p 0.78), time KS p 1.0, beta median +0.1 %, beta KS p 1e-3: at this sample size even two independent quarters of the reference disagree in beta at p 2e-7 or 0.39 depending on the split, so the beta KS is not yet a usable criterion |
| direct path v2 (2026-10-16) | MC 001400_00014516 with noise, odd events (11,074 events, 290 k charge clusters); same with noise off | joint resampling: a test cluster takes the whole response of one training cluster in its (Q, prompt fraction) bin (pulse count and the first 3 pulses, later pulses relative to the first), with the FEB clock phase removed in training and one sampled TDC0 per (event, FEB) put back, as stock (time residual width 4.7 -> 2.9 ns once TDC0 is removed; FEB-correlated shifts kept). Pulse count 0/1/2/3: 11.46/54.91/33.44/0.19 % ref vs 11.49/55.12/33.21/0.18 %; threshold 6-8 PE 14.2 vs 13.1 %, 8-10 PE 66.0 vs 64.4 %. First pulse PE median +0.19 % (KS p 0.45, binned ratio within 0.7 % beyond 2 sigma), time KS p 1.0, beta +0.13 % (binned 0.8 %). Second pulse PE, spacing and beta within 1-4 % binned (KS p 0.13-0.33). **Downstream (coincidence finder on direct pulses vs on stock pulses, same events): 1.3059 vs 1.3091 coincidences/event (-0.25 %, V-R4 1 %), per sector type EX 9,787/9,786, T1 1,564/1,589, T2 2,285/2,294, taggers 825/828; cluster PE median 815.1 vs 815.2 (V-R4 2 %), binned ratio within 2 sigma; duration median 24.25 vs 24.11 ns.** Sampling 3 us/event (numpy). Electronic-noise pulses: none in this MC (all stock pulses belong to a charge cluster); KPP data needs a noise-pulse model (no charges there) |
| direct path: FEB clock phase knob (2026-10-18) | same MC | TDC0 is drawn per (event, FEB) uniform in [`--feb-phase-offset`, offset + `--feb-phase-spread`] (default [0, 12.5] ns = stock MC); a tuning parameter because data need not follow the MC jitter. Narrowing to [3, 9] ns: cluster duration median 24.1 -> 23.0 ns, T1 coincidences +4.5 %, cluster PE median +1.2 %: the knob moves the downstream observables at a measurable level |
| noise pulses (2026-10-18, `reco/noise.py`, `reco-noise`) | KPP data 124155_000001, fit on even events (23.7 k), test on odd (23.7 k) | a noise waveform has no pulse on another SiPM of its counter and none in another layer of its sector type within 20 ns (1.38 per event, 3 % pulses each; in MC 0.19 per event, all signal: subtracted per channel with `--mc`). Model: per-event count negative binomial (k = 10 from the overdispersion), channel from the per-channel rates, first-pulse time from the window histogram, content (pulse count, PE, spacing, beta, failed) resampled jointly from the data library, per channel where it has >= 200 waveforms. Held-out data: **1.368 vs 1.376 noise waveforms/event, variance 1.54 vs 1.57**, pulses per waveform 1.033 vs 1.031, per-channel rate ratio 0.95 median (5-95 %: 0.63-1.34, Poisson on ~70 counts), PE median 7.98 vs 7.96 (binned ratio 4.6 % beyond 2 sigma: the PE spectrum is per channel and pooling is the remaining approximation), beta 13.53 vs 13.48. **Coincidences: noise pulses never form one at the data operating point (8 PE, 3-4 layers): stock 18,407 = signal-only 18,407 = signal + model noise 18,407 in the test events.** Sampling 0.2 us/event. Defining noise by "no partner on the counter" alone keeps 1.58/event and loses 5 tagger coincidences: real single-ended hits, not noise |
| cost (CPU numpy, one GPVM core, same 22,095 MC events) | | waveform path 5.0 ms/event (synthesis 1.6 + pulse reco 3.4); direct path 0.16 ms/event (features 0.15 + sampling 0.005). Stock `MakeCrvRecoPulses` on this sample (PyROOT): 3.7 ms/event. The numpy reco is CPU parity, not a speed-up; the speed-up target (R1, R2) is the JAX twin on GPU |

### The 11 data waveforms where pulse finding differs (V-R2)

Each was re-run through the stock class (`stockfit`) alone, with the waveform before it, and after ~200 earlier
waveforms, and compared with the fast path and the art job (`crvfastsim.reco inspect`). All 11 contain a failed fit
or a residual within 1 ADC of a threshold; on waveforms without either, finding is identical on data as on MC.

| cause | waveforms | evidence |
|---|---|---|
| stock fit state carried between waveforms | 2 (digis 5, 2) | `MakeCrvRecoPulses` keeps one `TF1` for the job, and Minuit2 starts each fit from the previous fit's step sizes. Alone, the stock class gives exactly the fast result (two good pulses); after earlier waveforms it fails the first fit and returns one fallback pulse, as the art job did. The stock result depends on collection order |
| knife-edge rounding after subtraction | 3 (digis 33, 49, 25) | the fits agree to ~1e-4 relative (e.g. digi 49: A 1666.66 vs 1666.67, beta 15.0498 both), but `SubtractPulse` rounds the fit at each sample and the residual sits at 39.2-39.3 ADC against `minADCdifference` 40: one rounding flips and a 2-4 PE secondary pulse appears or not. Digi 25: the same at the 1 % limit-proximity rule of `FailedFit` (stock fails, fast passes, also run alone) |
| different residual after a failed fit | 5 (digis 78, 52, 29, 48, 51) | long tails (24-36 samples); every pulse agrees up to a failed fit, whose parameters (at a limit) differ between Minuit2 and the fast fit; the subtracted tail then yields one more or one fewer small (2-6 PE) failed pulse |
| saturated waveform | 1 (digi 14) | 11 samples at ADC 4095; both fail the first fit; the stock code finds no further peak, the fast path three more failed pulses |

Proposal for the threshold: count "finding identical" over waveforms with no failed fit and no residual within 1 ADC
of `minADCdifference`, and track the rest through the flag-agreement criterion (99.97 %). For the Offline PR of the
faster CPU fit: resetting the fit state per waveform makes the stock result order-independent; worth raising with the
CRV group whatever happens to the PR.

### GPU throughput, Polaris (2026-10-16, PBS 7755570, reservation queue R7728155)

`core/iri/jobs/crv_reco_bench_polaris.json`: one node, one `crvfastsim reco-bench` process per A100 (each sees one
device), 200,000 synthetic KPP-like events per GPU (4 M waveforms of 12 samples, `reco/synth.py`), batch 65,536,
float64, 3 timed passes after compilation. The JAX-vs-numpy test and the core tests passed on the GPU node first
(18 passed).

| quantity | value |
|---|---|
| events/s per A100 (4 GPUs, each its own process) | 18,768 / 18,893 / 18,826 / 18,812 (mean 18,825) |
| per node | ~75,300 events/s |
| per event | 53 us (compile 2.0 s per process) |
| vs the same code on a laptop CPU (8 threads) | ~38x |
| vs stock `CrvRecoPulsesFinder` on one core (KPP data 3.73 ms/event = 268 events/s) | ~70x per GPU |
| goal R1 (>= 20,000 events/s per GPU, pulses + coincidences) | **94 % of the target, pulses only**: the coincidence finder is not ported yet |

**3 kernel rounds** (`--rounds 3`; waveforms with a pulse in the third round are redone by the numpy path; PBS 7755621,
same reservation and node layout): **49,566 / 49,362 / 49,290 / 49,296 events/s per A100 (mean 49,380; 20.2 us/event;
~197,500 per node)**, 2.6x the 8-round kernel and ~185x stock `CrvRecoPulsesFinder` on one core. On the KPP data
reference V-R2 still passes with 3 rounds (5,000 events: clean finding identical, flag agreement 99.976 %), with 155 of
103,080 waveforms (0.15 %) sent to the numpy path; in the synthetic bench none reached the third round. **R1 for the
pulse reco: 2.5x the 20,000 target**; the coincidence finder (the other half of R1) is next.

### V-R3: coincidence finder (2026-10-16)

`crvfastsim/reco/coinc.py` ports `CrvCoincidenceFinder` (hit building with the fiber-length time correction and dead-
neighbour PE compensation, `filterHits`, the order-dependent greedy `findClusters`, `checkCoincidence` over 2/3/4-layer
combinations with the time, slope and slope-difference cuts and the max-distance widening, the second clustering
across readout sides, `clusterProperties`), with the `epilog_extracted_v04` cuts for EX, T1, T2 and M1-M8. It is the
numpy reference (Python loops over the few hits of an event; events are the parallel axis for a later JAX twin).
`crvfastsim reco-vr3` runs it on the stock pulses of a reference table and compares with the stock clusters.

| sample | result |
|---|---|
| KPP data 124155_000001, all 45,066 events with pulses, 36,336 clusters | **pass: identical cluster lists in every event** (same sector type, same pulse set); PE, start and end times identical, slope within 6e-8, hit time within 0.03 ns (float rounding at t ~ 76 us) |
| cost (laptop, Python reference) | 0.72 ms/event (stock: 0.05-0.17 ms/event); a jitted twin is needed for R1, not for correctness |

### V-R3 with the JAX coincidence finder (2026-10-16)

`crvfastsim/reco/coinc_jax.py`: one jitted fixed-shape kernel per sector type over rows of (event, sector type) hit
groups padded to H. Fixed-shape equivalences: stock `findClusters` grows a cluster until nothing joins and its link test
is symmetric, so a cluster is the connected component of its seed (boolean transitive closure by log2 H squarings);
`checkCoincidence` is a dense tensor over K filtered hits per (side, layer) slot, and the max-distance widening is a
max over passing combinations (order-independent; the stable x-sort is reproduced with ranks). Only n = coincidence
layers combinations are checked (cuts are uniform per sector type, asserted). Groups run in the smallest tier
(H, K) = (16,6), (32,8), (64,12), (128,16), (256,24) they fit and are promoted when a (side, layer) slot overflows;
beyond the last tier (54 of 45,066 KPP events, up to 1,650 pulses) the event goes to `coinc.find_event(...,
vectorized=True)`, a numpy-broadcast combination check (same result as the scalar port; 15x faster on those events).

| check | result |
|---|---|
| `reco-vr3 --jax`, KPP data 124155_000001, 45,066 events | **pass: identical cluster lists in every event** (36,336 clusters); PE within 2e-3, start/end identical, slope 2e-7, avgtime 0.03 ns |
| unit test (synthetic tracks + noise + showers, small tiers so promotion and fallback run) | JAX == scalar numpy == vectorized numpy |
| CPU throughput (laptop, incl. host packing and fallback) | KPP data 8,200 events/s (kernel 3.6 s, fallback 1.6 s, packing 0.2 s per 45 k events); synthetic (`synth.coinc_events`, 25 pulses and 1 cluster per event) 9,100 events/s |

**GPU (PBS 7755999, preemptable queue, one process per A100, 100 k synthetic events each):** 12,070-12,860
events/s per process end to end. The jitted kernel is 1.1-1.2 s per 100 k events (**~85,000 events/s per A100**);
the host packing (0.7 s) and the numpy fallback for 0.2 % of events (5.7-6.2 s) dominate. R1 (pulses + coincidences
>= 20,000 events/s per A100): kernels 20 us + 12 us = 32 us/event -> 31,000 events/s; end to end 20 us + 80 us ->
10,000 events/s until the fallback leaves the critical path (a larger last tier on GPU, or the fallback on spare CPU
cores in parallel with the next GPU batch).

**With the exact block split (PBS 7760400, same bench):** 19,100-19,800 events/s per A100 end to end (51 us/event):
kernel 2.0-2.1 s, host splitting ~2.3 s, packing 0.5 s, numpy fallback 0.2-0.4 s (4-10 of 100 k events) per 100 k
events. Pulses + coincidences: 20 + 51 us -> ~14,000 events/s end to end (kernels alone 20 + 21 us -> ~24,000).
The remaining gap to R1 is host work (split + pack, ~28 us/event), which can overlap the GPU or move onto it.

### V-R1 with the JAX waveform synthesis (2026-10-16)

`crvfastsim/reco/waveform_jax.py`: the waveform synthesis as one jitted fixed-shape function (host packing of the stock
charge clusters into [C, Q] matrices; on device a [C, Q, L] template gather and sum, noise, the 5-sample zero-suppression
window and a lax.scan for the 12-sample digi split; ADC conversion). Against the numpy port on synthetic clusters:
sample-identical ADCs and voltages within 1e-12 (unit test). Against the stock noise-free reference (`reco-vr1 --jax`,
5,000 MC events): **all 128,718 stock digis reproduced sample-identically**, the same 7 extra digis as the numpy path
(FEBs without a stock digi in the event, TDC0 unknown to the test). CPU (laptop) synthetic throughput: ~42,000
events/s.

**GPU (PBS 7755673, reservation R7728155, one process per A100):** waveform synthesis **1.148-1.154 M events/s per A100**
(0.87 us/event; 20 SiPM clusters of ~28 PE per event, noise on, compile 0.75 s), i.e. **R2 met by a wide margin**
(the stock `CrvWaveforms` + `CrvDigi` are part of the 19.8 ms/event CRV response on one core); pulse reco in the same
job 49.4-49.6 k events/s per A100 (3 rounds), reproducing PBS 7755621. All 21 GPU-node tests pass, including JAX-vs-
numpy for both kernels. The job fetches the one public Offline data file the synthesis needs
(`CRVResponse/data/singlePEWaveform_v3.txt`, from GitHub) onto Eagle, since Polaris has no CVMFS; the first run
without it (7755663) failed that test only.


### V-R1 with noise on (2026-10-19)

`reco-vr1 --noise [--jax] --nonoise-ref crvreco_mcnonoise...root crvreco_mc...root`: the fast arm adds its own
Gaussian noise (0.4 mV per sample; JAX key or numpy rng) to the same stock charges with the reference TDC0s and is
compared with the stock chain run with noise on (`crvreco_mc`, same events, charges and TDC0 as the noise-free table).
Noise is 0.92 ADC rms against signals of hundreds of ADC, so a per-channel ADC KS (the V-R1 criterion) mostly tests the
shared signal; the noise model itself is tested by the residual of each noisy arm against the stock noise-free digi
(digis present in all three tables).

| check (MC 001400_00014516, 22,095 events) | JAX | numpy |
|---|---|---|
| digis (stock 565,639) | 565,655 | 565,615 |
| per-channel ADC KS, 1,144 channels with >= 200 digis | **min p 0.81**, 0 below 0.01 (pass) | min p 0.76, 0 below 0.01 |
| digi count per channel, fast/stock (5-95 %) | 0.995-1.005 | |
| residual vs noise-free stock, 6.7 M samples: rms (stock 0.9762; expected 0.964 = 0.92 (+) 1/sqrt(12)) | 0.9755 | 0.9760 |
| residual mean (ADC) | 0.0025 (stock 0.0027) | 0.0020 |
| residual KS vs stock residual | **p 0.85** | p 0.13 |
| neighbour-sample correlation (white noise) | -2e-5 (stock -7e-4) | 5e-4 |

**V-R1 passes with noise on** for both implementations: same noise width, shape and whiteness as stock, digi counts
per channel within 0.5 %. The +-16 digis are threshold crossings decided by the noise draw.

### R5 reproducibility (2026-10-19)

`reco-repro` runs the jitted stages on seeded synthetic inputs (waveform synthesis with noise from a JAX threefry key;
pulse reco, 3 rounds + numpy redo; the pipelined coincidence finder) and hashes each stage's output. Laptop CPU, 2,000
events: the same seed gives identical hashes in one process and across processes (`--against` a saved run), another
seed changes all three; unit test `test_repro_bit_identical_per_seed`. **Polaris (PBS 7761649), 5,000 events:** two GPU runs
bit-identical in all three stages; GPU vs CPU on the same node: **waveforms (with noise) sample-identical** (100,000
digis, threefry is backend independent), pulses all matched (99,948) with PE within 3e-9 relative and time within
6e-8 ns (98 % differ in the last bits: XLA fuses the fit arithmetic differently), **clusters: same 4,812 clusters per
event and sector type**, PE within 2e-7. KS p 0.60 / 0.10 (pulse PE / time), 0.88 (cluster PE). The run reported
`pass_statistical: false` only because `match.ks_2samp` returned p = 0 for identical samples (the alternating
Kolmogorov series does not converge at small lambda); fixed with the small-lambda form (agrees with scipy). **R5 met:**
bit-identical per seed and backend, CPU and GPU equal to float rounding.

### R1 measured: pulses -> coincidences in one pipelined pass (2026-10-19, PBS 7761649)

`reco-bench synthetic --stage chain` (chain_jax: chunked; pulse reco of chunk k on the device while the host collects
chunk k-1's pulses and launches its coincidence finding and finishes chunk k-2), 200,000 synthetic events per A100
(25 waveforms per event, ~1 coincidence per event), 3 pulse rounds, 8 chunks, one process per GPU:

| GPU | events/s | us/event |
|---|---|---|
| 0 | 22,146 | 45.2 |
| 1 | 22,622 | 44.2 |
| 2 | 23,020 | 43.4 |
| 3 | 20,104 | 49.7 |

**R1 met: 20,100-23,000 events/s per A100 end to end** (>= 20,000), including the numpy redo of exhausted pulse fits
and the numpy coincidence fallback (1-8 of 200 k events). GPU 3 was also slower in the pulse-only and waveform stages of
the same job (41 k and 1.06 M vs 49 k and 1.15 M), a node effect. Coincidence stage alone in the same job: 51,800-56,100
events/s per A100 (was 19,100-19,800 before async dispatch and the 64-hit split threshold). Host time per 200 k events:
pulse launch 5.2 s, pulse collect 0.3 s, coincidence prepare+launch 1.8 s, coincidence finish 1.6 s of a 8.7-9.9 s
wall: the remaining headroom is in overlapping the pulse launch (host batching) with the coincidence host work.
