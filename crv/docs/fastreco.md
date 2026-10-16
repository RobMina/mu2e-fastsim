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

The kernel always runs `max_pulses` = 8 pulse rounds with the full fit, while 99.8 % of data waveforms have at most two
pulses: the obvious next step is fewer rounds (e.g. 3) with the waveforms still active after the last round sent to the
numpy path, then batch-size and LM-iteration tuning.
