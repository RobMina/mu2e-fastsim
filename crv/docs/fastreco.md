# Fast CRV reconstruction (stage 1): goals, validation, prototype

Status: **goals and thresholds agreed 2026-10-16.** Prototyping both paths (below) on existing digis.

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
- `crvfastsim/reco/`: `python3 -m crvfastsim.reco vr2|vr2stock|vr1|direct|cost REF.root ...` (README section 10).
- Stock cost for scale (TimeTracker means per event): `CrvRecoPulsesFinder` 1.6 ms (extracted MC) and 3.7 ms (KPP
  data), 97 % of it in `TGraph::Fit`; the CRV response 19.8 ms per accepted event (`plan.md` section 3).

## Results

| check | sample | result |
|---|---|---|
| unit tests | synthetic waveforms | fit recovers injected pulses; peak finder, pedestal rule, digi grouping, waveform linearity, ZS |
| V-R2, V-R1, direct, cost | needs the reference tables (build of the `CrvFastSim` art package) | not run yet |
