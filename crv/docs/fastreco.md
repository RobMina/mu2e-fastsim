# Fast CRV reconstruction (stage 1): goals, validation, prototype

**Thread B continuation (2026-10-26):** the historical results below do not certify the newer stock backing.
The missing opposite-side attachment stage is now ported and passes full-finder membership closure on retained
shards 11–12 (all 5,875 clusters, 1,254 restored links). Numerical X/time properties, GPU validation and downstream
DQM remain open; see [the candidate port and validation ledger](thread-b-membership-port.md).

Status: **goals and thresholds agreed 2026-10-16.** First comparisons on existing digis done 2026-10-16 (Results).
As of 2026-10-22: V-R1 to V-R5 pass, and R1, R2, R3, R4 and R5 are met (R3 and V-R5 at the end of Results); R6 is not
formally assessed. Newest: the out-of-time background model (last section).

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

### Tuning knobs as runtime data (2026-10-19)

A Mu2eBO point that changes a reco or response setting no longer recompiles the jitted kernels (a recompile is
40-50 s per point on Polaris). Only settings that fix array shapes stay compile-time:

| kernel | compile-time | data (any value reuses the compiled kernel) |
|---|---|---|
| `pulsefit_jax.reconstruct_dense` | `max_pulses` (rounds), `samples_before`/`samples_after` (window), `lm_iterations` | minADCdifference, default/min/max beta, max time difference, pulse-height ratios, LE-time factor, pedestal undershoot threshold, failed-fit limit tolerance (`pulsefit_jax.KNOBS`) |
| `waveform_jax.synth_dense` | `L` (samples), `zs` (digi length) | noise, minVoltage, ADC conversion, pedestal, ADC range, 1PE template and its precision |
| `coinc_jax.find_dense` | length axis, which combination sizes can occur, K (slots), whether the sector type has one set of cuts | every `sectorConfig` cut **per sector** (PE threshold, adjacent and coincidence time windows, slopes, slope difference, coincidence layers, minClusterPEs, initial/final max distance); fibre speed, max time interval, time offset |

- **Per-sector cuts.** The cuts travel with each hit, so sectors of one type may differ (Run 2 per-sector thresholds);
  the finder applies them as `CrvCoincidenceFinder` does (time window, slope limits and slope difference of a
  combination from its hits' sectors; the layers checked per cluster from the min and max `coincidenceLayers` of its
  hits; minClusterPEs per cluster). Types whose sectors share their cuts take a scalar path (same result, no cost:
  50 k synthetic events 9,300-9,500 events/s on the laptop CPU before and after).
- **Identity at the default settings:** waveforms (with and without noise) and every discrete output are bitwise
  identical to the previous kernels; pulse PE, time and beta move at the last-bit level (max 1.3e-9 relative: XLA
  folds traced constants differently from compile-time ones); V-R3 identical cluster lists on KPP data (45,066
  events) and MC (22,032); V-R2 (5,000 data events, 3 rounds) clean finding identical.
- Tests: `test_knobs_are_runtime_data_no_recompile` (sweeps of 8 pulse-reco, 3 waveform and 3 coincidence settings
  compile each kernel once, and the outputs follow the knobs); `test_coinc_jax_per_sector_cuts` (mixed 3/4-layer
  taggers, per-sector thresholds, slopes, distances and minClusterPEs: JAX equals the numpy port).

### R4 container (2026-10-20)

`container/` in the umbrella (definition, `build.sh`, `fetch_offline_data.sh`, README): built from git-tracked files
only (`git archive` of the umbrella and submodules at their checked-out commits, recorded in
`/opt/mu2e-fastsim/VERSION`), the two public Offline files under `/opt/offline_data`, no CVMFS, no Mu2e-internal data.

- **Polaris (PBS 7763332, apptainer 1.3.2 from `/soft/spack/base/0.8.1/install/modulefiles/Core`):** image built on
  the compute node in 109 s (fakeroot works); **cold start 10.7 s** (2.3 s copy to node-local disk + 8.4 s first exec
  with an empty cache; goal <= 5 min); all 4 A100s visible; JAX GPU tests pass inside the image. The first chain bench inside
  it failed on a packaging bug (the subpackage lists in `pyproject.toml` missed `crvfastsim.response` and
  `trkfastsim.analyses`; source-tree runs never notice); fixed with automatic package discovery. **Rerun (PBS
  7763444): pulses -> coincidences chain inside the container 20,918 events/s on one A100** (47.8 us/event; venv
  20,100-23,000), cold start again 10.7 s.
- **Size:** the GPU image is 3.5 GB (5.6 GB unpacked, 4.8 GB of it the pip CUDA wheels). R4's target is "the fast-sim
  image + < 0.5 GB" (decision 2026-10-20: read as the reco's increment on the shared JAX/CUDA image, ~1 MB, within the
  fast sim's own P5 limit of 5 GB), so **R4 is met on size and cold start**. To make the image cheaper to move anyway:
  `build.sh` now drops the CUDA wheels no kernel loads (cuDNN 1.2 GB, NCCL 0.47 GB, NVSHMEM 0.40 GB, cuFFT 0.28 GB
  unpacked; `TRIM`, override with `TRIM=""`) and has a CPU-only variant (`VARIANT=cpu`) for FermiGrid, local runs, CI
  and CPU endpoints (veto plan E8; unpacked tree 0.8 GB). Trimmed vs full GPU image on Polaris: PBS 7763477 (pending);
  CPU image: building on mu2egpvm06.

### V-R4: fast chain vs stock chain from the same CrvSteps (2026-10-20)

`crvfastsim reco-vr4` (`reco/vr4.py`, batch `scripts/run_vr4.sh`): each G4 CrvStep table of the g4ref_v1 shards
(`crvsteps.<shard>.root`, the steps of the G4 dts file) runs through the fast chain (`crvfastsim/chain.py`: fast photons
-> fast SiPM charges -> fast waveforms/digis with noise and one TDC0 per (event, FEB) -> pulses -> coincidences;
SiPM-status mask and CRVCalib constants from the reference job's conditions, Sim_best v1_5 run 1400) and is compared
with the stock art chain on the same dts file (`ref.<shard>.root` of `scripts/run_v2_arms.sh`). Accumulators and checks
are v2compare's, per sector group, with the V-R4 thresholds: PE per seq-0 pulse median within 2 %, binned PE spectrum
ratio within +-3 % after 2 sigma, coincidence clusters per event within 1 % after 2 sigma, bar-rate chi2/ndf.

**All 20 shards (50,008 events, numpy chain on mu2egpvm06, 30 s per shard): V-R4 passes in every group.**

| group | seq-0 pulses stock / fast | PE median stock -> fast | binned PE ratio, max dev (raw) | clusters stock / fast |
|---|---|---|---|---|
| EX | 739,064 / 739,194 | 46.47 -> 46.48 (+0.02 %) | 0.9 % | 39,363 / 39,369 (+0.02 %) |
| T1 | 75,078 / 75,134 | 76.25 -> 76.19 (-0.08 %) | 4.0 % (0 after 2 sigma) | 6,763 / 6,759 (-0.06 %) |
| T2 | 197,640 / 197,622 | 47.69 -> 47.71 (+0.05 %) | 2.3 % | 9,424 / 9,422 (-0.02 %) |
| taggers | 35,272 / 35,286 | 76.13 -> 76.16 (+0.04 %) | 5.5 % (0 after 2 sigma) | 3,624 / 3,628 (+0.11 %) |

- Bar rates (events with a >= 5 PE pulse per bar, 576 bars): chi2/ndf 0.01, max pull 0.4. Both arms see the same
  muon crossings, so the counts are strongly correlated and the independent-sample errors used for the 2-sigma
  allowances are conservative; the raw differences above are the measure.
- **JAX chain, all 20 shards (FNAL grid, 2026-10-20): V-R4 passes in every group** (EX PE median 46.47 -> 46.47,
  T1 76.25 -> 76.05 (-0.26 %), T2 47.69 -> 47.69, taggers 76.13 -> 76.17; clusters per event within 0.2 % in every
  group; bar rates chi2/ndf 0.01). Grid jobs (`grid/submit_vr4.sh`, `grid/grid_vr4.sh`): one shard per job, the CPU
  container unpacked to a sandbox on the worker (grid jobs run inside the worker's container without /dev/fuse, so a
  SIF cannot be mounted there), the committed core+crv code shipped as a 285 kB tarball, inputs and outputs via
  pnfs scratch. 10.8 min and 5.7 GB per shard on a grid node; 3 of 20 shards exceeded 8 GB with 250-event batches
  and reran with `--batch-events 100` and 12 GB. Shard 1 on the grid equals the same shard run on a GPVM bit for bit.
  Memory-hungry runs go to the grid, not to the GPVMs.
- Not covered yet: response knobs away from the nominal point. (The EX layer-efficiency row was covered on 2026-10-21:
  next section.)
- The JAX pulse path (`chain_jax.run`) now adds the per-channel DB time offset (`W["toff"]`), as `CrvRecoPulsesFinder`
  and the numpy path do; earlier callers passed no offsets (test `test_chain_jax_adds_db_time_offsets`).
- Cost of the numpy chain per event with CRV steps (mu2egpvm06, one core): photons 3.4, SiPM 1.6, waveforms 0.5,
  pulses 1.7, coincidences 0.7 ms.

### V-R4 EX layer efficiency with extrapolated tracks (2026-10-21)

`crvfastsim reco-vr4eff` (`reco/vr4eff.py`; grid `grid/submit_vr4eff.sh`, `grid/grid_vr4eff.sh`). Sample: the
timing-fixed CRV-only MC `crvonly_v05` (CRVTime applied with scale -1, one FEB clock phase per event). Stock arm: its
EventNtuples (`crvpulses`); fast arm: the fast chain (JAX, CPU container on the grid) on the CrvSteps of the same
production dts files, with the v05 settings (common FEB phase, CRVTime net zero). Both arms use the same quality
KKLine track per event (`crvextrap.trackline`, from the tracker; detector frame converted to Mu2e coordinates). Per EX
layer: denominator = tracks crossing the layer inside the sector; hit = seq-0 pulse >= 8 PE within one counter of the
predicted one and within +-20 ns of the track time + 54.4 ns (offset measured on the stock arm). The difference error
comes from the discordant tracks (same tracks in both arms).

**19 of 20 files (prep on a GPVM in seconds; fast arm 1.6-4 h per file on the grid): passes in every layer.**

| EX layer | tracks | stock | fast | fast - stock (points) |
|---|---|---|---|---|
| 0 | 2,179 | 81.28 % | 81.41 % | +0.14 +- 0.10 |
| 1 | 2,170 | 81.24 % | 81.20 % | -0.05 +- 0.05 |
| 2 | 2,154 | 80.97 % | 81.01 % | +0.05 +- 0.08 |
| 3 | 2,140 | 80.05 % | 80.09 % | +0.05 +- 0.08 |

The ~81 % level is the track-pointing resolution (~24 mm at EX, as in `crvextrap exdenom`), common to both arms. The
last file (`001400_00001023`) exceeded the 4 h lifetime and reruns with 50-event batches and 8 h (cluster 30742793).
With this row, **V-R4 is complete**.

### R3 driver and the response objective (2026-10-21)

`crvfastsim reco-r3` (`reco/r3.py`): cached G4 CrvStep tables (e.g. `prod_as_tables/crvsteps.*.root`, pdg joined at
load) replayed with fresh seeds until `--target` events with CRV steps (the R3 unit), through response (JAX: fused
device response; numpy: `chain.charges`), waveforms, grouping, pulses + clusters and, with `--objective`, the
`crvextrap crvresponse` histograms. One process per accelerator; `--merge` sums a node's JSONs (`--wall` = job wall)
and gives the minutes for 1 M events. `--v05` applies the `crvonly_v05` digitisation settings. Pass 0 includes JIT.

The objective path: `chain.clusters(..., detail=True)` (JAX: `chain_jax.run(keep_pulses=True, with_pos=True)`;
`coinc_jax` returns `cl_pos` only when asked, so the default R1 path is unchanged) -> `reco/ntview.py` (crvpulses /
crvcoincs jagged arrays with EventNtuple branch names, detector frame, `crvHitIndex` links) ->
`crvextrap.crvline.fit_arrays` (CRV-only line) -> `crvextrap/analyses/crvresponse.py` (plan §2.3 observables; objective
= mean over terms of shape chi2/ndf, plus a relative-tolerance term for the out-of-time rate). Tests: JAX positions
equal `coinc.py` per cluster; `ntview` links and member sums.

First numbers (CPU only; GPU pending):

| run | events with steps | ms/event (response / digis / reco / view + objective) |
|---|---|---|
| numpy, mu2egpvm04, `crvsteps.001400_00000990` (v05 settings) | 22,573 | 6.61 / 0.64 / 4.21 / 0.16 |
| JAX on the laptop CPU, g4ref shard 11, 3 passes | 6,633 | 12.4 / 3.2 / 6.5 / 0.12 |

- **Host glue is negligible (A3).** On the GPVM numpy chain: pdg join 0.007, TDC0 draw 0.001, digi grouping 0.008,
  `ntview` 0.010, line fit + histograms 0.15 ms/event. In the JAX CPU profile, Python outside JAX kernels is 1.5
  ms/event of 22; the rest is kernel time and compilation. The R3 risk is device time (response, then reco), not glue.
- **Objective closes fast vs stock on the same steps:** fast (numpy, v05 settings) vs the stock v05 EventNtuple of the
  same dts file: objective 0.90 over 32 terms (largest term 1.7, tagger out-of-time rate); lines 19,863 vs 19,862 (z),
  3,763 vs 3,763 (x). Fast JAX vs fast numpy on shard 11: 0.87.
- **Stock v05 MC vs data (124336, 200 k events):** objective 178. Largest: out-of-time rates 8-19x lower in MC than in
  data, `dt_cluster_T2` and `dt_cluster_T1` (data timing tails), `pe_layer_T1`. This is the MC-data gap the response
  fit is meant to close; the term weights and the out-of-time definition need a review before BO uses it.

### R3 met on Polaris (2026-10-22)

`reco-r3` on one Polaris node (4 A100, PBS 7771210, `--v05`, 20 MDC2025as step tables replayed with fresh seeds): **1 M
events with CRV steps end to end (response + waveforms + reco + objective) in a 624 s job wall = 10.4 min** (goal
<= 15 min, stretch 3). Warm throughput 3,161 events/s per node (790 per A100); per A100 warm ms/event: response 1.45,
digis 0.30, reco 0.34, objective 0.06. JIT compilation (pass 0) takes ~370 s of the ~570 s per process, so a small-N R3
run measures compilation, not throughput. The fast chain on GPU still agrees with stock on the same steps (objective
1.2 against the stock v05 EventNtuple of the same file).

### Response-point scoring contract (2026-10-25)

`kit_task.score_result` now requires the frozen 32-term `kit_task.TERMS` manifest, with P3's explicit
`dt_cluster_T2` exclusion giving 31 scored terms. Missing, empty/zero-ndf, nonfinite or newly appearing terms reject
an objective point instead of changing its denominator. Histogram counts must be finite/nonnegative and candidate
and reference shapes must agree. The manifest and exclusions are persisted in `objective.json`. If an excluded
term is unavailable, `objective_all` is unavailable rather than an average over a different set; requesting that
metric downstream still requires it to be finite. The diagnostic `objective(..., required=None)` remains available
for exploratory partial histograms; it is not the kit scoring route.

Scoring also refuses incomplete event targets, absent generated-event exposure and recorded residual response
overflow. R3 raises before consuming fused output with residual photon/SiPM overflow (after the existing device
retry), and rejects nonzero thermal/trap settings and windows other than extracted `(0, 100000)` ns on both R3
backends. These guards are specific to R3 and its scoring route, not a claim of complete general-chain/backend
admission validation. Numerical capacities and stock-compatibility switches are not new response knobs.

Synthetic coverage is in `tests/test_response_contract.py`, including the rendered scoring block and injected
residual overflow before host consumption. Existing complete synthetic scores are unchanged. CPU tests are not a
GPU physics validation: independently seeded nominal/best points on held-out input shards, covariance-aware
uncertainty checks and stock closure remain required. In the current Mu2eBO propagation fixture the surrogate
noise is fixed at **0.7**; `objective_sigma` is only an extra metric, not row-wise GP noise. Do not combine changed
reco outputs with the old pilot board without establishing measurement compatibility.

### Size-free objective (2026-10-22)

The objective now lives in this package as `crvfastsim.objective` (`crvline`, `crvresponse`, `bars`; numpy/awkward
only; `python -m crvfastsim.objective`), so the fast chain, the grid jobs and Mu2eBO import one copy. The raw shape
chi2/ndf grew with the sample size for a fixed MC-data difference, which a BO loop cannot use. Each histogram term is
now the bias-corrected triangular discrimination of the two shapes, on the chi2/ndf scale at N0 = 10 k entries, with
`sigma` (Poisson resampling of the candidate: the BO `obs_noise`) and `sigma_ref` (the reference's own noise). The raw
value is kept as `objective_raw`. Check: the same comparison at 22 k and 1 M events now agrees within its errors.
`reco-r3 --knob NAME=VALUE` (`chain.KNOBS`) and `R3_KNOBFILE` in `r3_polaris.sh` (one point per GPU) run a knob point.

### V-R5: the objective from both paths on a knob scan (2026-10-22)

Stock arm on the FermiGrid (`grid/submit_vr5_stock.sh`, points in `grid/vr5_points.txt`), fast arm through
`reco-r3 --knob`, both on the same five MDC2025as files, scored against one reference (`grid/vr5_collect.sh`,
`grid/vr5_verdict.py`). Knobs: photon yield (0.85-1.12), crosstalk (0.02-0.08), single-PE waveform stretch (0.95-1.35;
`CrvWaveforms.singlePEWaveformStretchFactor`, which replaced the time constant because that only shifts the pulse).
**Pass:** the same minimum from both paths for every knob, and every fast - stock difference (largest 0.27, 0.49,
0.27) within 1/10 of the stock spread over the scanned range (3.8, 11.6) where that spread exceeds the noise of a
difference, and within noise otherwise (crosstalk is flat within noise).

One fix was needed first: the fast arm replays only events with CRV steps, so its per-event out-of-time rates were
~10 % high and every fast point sat below stock. `reco-r3` now adds the source files' events without CRV steps to the
exposure (`events_without_steps`, `crvresponse.add_empty_events`), as a stock EventNtuple holds every event. A stock
run with thermal noise on (ThermalRate 1e-4) gives the same out-of-time rates and objective as nominal, so thermal
noise is not the missing out-of-time source.

### Out-of-time background model (2026-10-22)

The out-of-time CRV pulses in KPP data are real single-bar light (most have the partner SiPM within 20 ns and come one
bar at a time, flat in the window), which the cosmic MC makes at only ~10 % of the data rate, and none of the V-R5 knobs
moves them. `crvfastsim.reco.background` adds them to the fast chain: groups of out-of-time pulses (per event and
sector, gaps <= 30 ns) resampled from a library built from data, at the net rate data - MC, with a negative-binomial
count per event. They are merged with the signal pulses before the coincidence finder (the `add_pulses` hook in
`chain_jax`, and the numpy path); events without waveforms are filled directly. Knob `background_scale` (fast chain
only). Library and rate: `reco-background --emit/--merge` (per-file partials; GPVM scatter `scripts/run_background_fit.sh`).
Check on CPU (`scripts/background_check.py`): the out-of-time rates come within a few % of the reference, the
out-of-time terms of the objective drop to shape level, and the coincidence count is unchanged (30,138 vs 30,132).
Tests: `tests/test_background.py`.
