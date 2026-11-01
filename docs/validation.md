# Validation

Summary of what each stage is validated against. The full tables (sample, metric, result, threshold, verdict, date,
commit) are in each repository's `docs/validation.md`: [CRV](https://github.com/RobMina/mu2e-crv-fastsim/blob/main/docs/validation.md),
[tracker](https://github.com/RobMina/mu2e-tracker-fastsim/blob/main/docs/validation.md),
[core](https://github.com/RobMina/mu2e-fastsim-core/blob/main/docs/validation.md); the beam stage is below. Thresholds
were agreed before the comparisons were run. What is open or not validated: [limitations.md](limitations.md).

## Summary

| Stage | Check | Result | Verdict |
|---|---|---|---|
| CRV | V0 geometry; V1 muon crossings vs Geant4 (extracted, Run 1A) | 576 bars to 3.6e-12 mm; crossings within 0.9 % per sector group | pass |
| CRV | V1 Run 2 geometry | below-plane scatter-in missing | **fail** |
| CRV | V4 photons and SiPM charges vs stock (CPU and GPU) | within 0.5 % and 0.2 ns | pass |
| CRV | V-R1..V-R5 waveforms, pulses, coincidences, end to end, objective | identical or within thresholds | pass |
| CRV | default chain vs Geant4 + stock, 5 M held-out events | 21/155 histograms flagged (steps), 13/133 end to end, 0/133 fast vs stock response+reco on the same steps; counts within 0.25 % | open |
| tracker | T-V0 geometry; T-V1 primary-muon steps vs Geant4 | 0 steps outside their straw; every T-V1 threshold | pass |
| tracker | digis, hits, clusters, seeds vs stock on the same input | within the reseed floor; identical | pass |
| tracker | F1 line fit vs stock KKLine | acceptance agreement 95.6 % (target 99.5 %) | open |
| tracker | T-V3 reco yield on fast steps vs Geant4 + stock | -2.04 +- 0.37 % (threshold 1 %) | **fail** |
| tracker | any-primary-muon rate | -0.89 +- 0.32 % | open |
| beam | stop fraction, nominal target, vs Geant4 on the same muons | ratio 0.9976 +- 0.0058 | pass |
| beam | stop fraction, 49-foil geometry move | ratio 0.9950 +- 0.0031 | pass |
| core | JAX loss sampler vs NumPy; device CRY vs stock CRY | within criteria; accepted rate 0.9998 +- 0.0016 | pass |
| electrons (branch) | dE/dx vs ESTAR; thin layers vs Geant4 | 0.6 % / 1 %; chi2/ndf 0.6-1.6 | pass, not on `main` |
| electrons (branch) | 1 X0 tungsten; positrons | no multiple scattering, no annihilation | **fail**, not on `main` |

## Beam stage

Reference: the stock DS stage of SimJob MDC2025ac (`Production/JobConfig/pileup/MuBeamResampler.fcl`) run with a
per-muon tree (`beam/fcl/g4ref_mubeam_ntpart.fcl`) on the same MuBeamCat muons as the fast arm. "Commit" is this
repository's `main` on that date.

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| stop fraction, nominal target (37 foils) | Geant4 DS stage | 105,350 MDC2025ac muons | target stop fraction; stops per foil; stop radius | 0.35834 vs 0.35919, ratio 0.9976 +- 0.0058; per-foil chi2/ndf 0.80; radius KS p 0.70 | ratio within 2 sigma of 1 | pass (the nominal reference is partly from an incomplete job; certification pending) | 2026-10-23 | `c6d734c` |
| stop fraction, 6 + 6 extra foils (49) | Geant4 DS stage, same muons | 105,350 muons | stop fraction; change vs nominal; stops per foil | 0.66588 vs 0.66925, ratio 0.9950 +- 0.0031 (1.6 sigma); x1.858 vs x1.863; per-foil chi2/ndf 2.51 (first, thicker foil 15 % low) | ratio within 2 sigma | pass on the fraction | 2026-10-23 | `c6d734c` |
| loss model at the stopping edge, vs finer Geant4 steps | Geant4 with refined steps | slab tests | edge-variance ratios | 0.89 / 1.02 against refined steps vs 1.23 / 1.24 against stock steps | equivalence band | indeterminate | 2026-10-30 | `36176d8` |

Throughput: 14,260 muons/s on one A100 MIG slice, 42,270 on a whole A100, 184 on one CPU core; Geant4 83 muons/s on
one GPVM core.

## Device precision (pending change)

Every result above uses float64 on the device. A float32 device mode is validated on 100 CRV shards (within the
float64 seed-to-seed spread, pulse PE about -0.1 %) but is not on `main`.
