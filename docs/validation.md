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
| beam | stop fraction under much finer Geant4 steps, same muons | refined / stock 1.0016 +- 0.0020 nominal, 1.0019 +- 0.0011 moved | pass |
| beam | stop fraction vs refined-step Geant4 | ratio 0.9927 +- 0.0058 nominal; 0.9934 +- 0.0031 moved (2.15 sigma) | pass nominal, **fail** moved |
| core | JAX loss sampler vs NumPy; device CRY vs stock CRY | within criteria; accepted rate 0.9998 +- 0.0016 | pass |
| electrons (branch) | dE/dx vs ESTAR; thin layers vs Geant4 | 0.6 % / 1 %; chi2/ndf 0.6-1.6 | pass, not on `main` |
| electrons (branch) | 1 X0 tungsten; positrons | no multiple scattering, no annihilation | **fail**, not on `main` |

## Beam stage

Reference: the stock DS stage of SimJob MDC2025ac (`Production/JobConfig/pileup/MuBeamResampler.fcl`) run with a
per-muon tree (`beam/fcl/g4ref_mubeam_ntpart.fcl`) on the same MuBeamCat muons as the fast arm. "Commit" is this
repository's `main` on that date.

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| stop fraction, nominal target (37 foils) | Geant4 DS stage | 105,350 MDC2025ac muons | target stop fraction; stops per foil; stop radius | 0.35834 vs 0.35919, ratio 0.9976 +- 0.0058; per-foil chi2/ndf 0.80; radius KS p 0.70 | ratio within 2 sigma of 1 | pass (first reference, from a job that stopped at event 105,100; clean rerun below) | 2026-10-23 | `c6d734c` |
| stop fraction, 6 + 6 extra foils (49) | Geant4 DS stage, same muons | 105,350 muons | stop fraction; change vs nominal; stops per foil | 0.66588 vs 0.66925, ratio 0.9950 +- 0.0031 (1.6 sigma); x1.858 vs x1.863; per-foil chi2/ndf 2.51 (first, thicker foil 15 % low) | ratio within 2 sigma | pass on the fraction | 2026-10-23 | `c6d734c` |
| stop fraction, nominal, clean stock rerun | Geant4 DS stage, new seed, job exits 0 | 105,350 muons (105,099 events) | stop fraction; stops per foil | 0.35834 vs 0.36038, ratio 0.9943 +- 0.0058; per-foil chi2/ndf 1.21; the two stock runs differ by 1.0033 +- 0.0020 | ratio within 2 sigma | pass | 2026-11-01 | `320b16c` |
| step sensitivity of the stop fraction, nominal | same stock vs Geant4 with refined steps, same muons | 105,350 muons | paired ratio refined / stock | 0.36096 vs 0.36038, 1.0016 +- 0.0020; per-foil chi2/ndf 0.62; first foil 1213 vs 1199 | abs(ratio - 1) + 3 SE <= 1 % (declared before the run) | pass (0.75 %) | 2026-11-01 | `320b16c` |
| step sensitivity of the stop fraction, 49 foils | saved stock vs Geant4 with refined steps, same muons | 105,250 muons | paired ratio refined / stock | 0.67052 vs 0.66925, 1.0019 +- 0.0011; per-foil chi2/ndf 1.14; first foil 4588 vs 4513 | abs(ratio - 1) + 3 SE <= 1 % (declared before the run) | pass (0.52 %) | 2026-11-01 | `320b16c` |
| stop fraction vs refined steps, nominal | Geant4 with refined steps | 105,350 muons | ratio; stops per foil | 0.35834 vs 0.36096, ratio 0.9927 +- 0.0058; per-foil chi2/ndf 1.07 | ratio within 2 sigma | pass | 2026-11-01 | `320b16c` |
| stop fraction vs refined steps, 49 foils | Geant4 with refined steps | 105,250 muons | ratio; stops per foil | 0.66612 vs 0.67052, ratio 0.9934 +- 0.0031 (2.15 sigma); per-foil chi2/ndf 3.76 | ratio within 2 sigma | **fail**, narrowly | 2026-11-01 | `320b16c` |
| parked slab kernel at the stopping edge | Geant4 slab, refined steps | 5,000 G4 / 8,000 fast per cell | edge-variance ratios | 0.89 / 1.02 against refined steps vs 1.23 / 1.24 against stock steps | 5 % at 3 SE | indeterminate; this kernel is on `park/beam-loss-studies`, not the default transport, and says nothing about `legacy` | 2026-10-30 | `36176d8` |

Refined steps: `/process/eLoss/StepFunctionMuHad 0.005 0.00005 mm` and `/process/em/binsPerDecade 168`, set through
`physics.producers.g4run.g4Macro`, same release (SimJob MDC2025ac) and muons. In a foil at the stopping edge this takes
a muon from a median of 3 steps to about 500. One refined level is not a convergence proof, but the fraction does not
move: the two geometries combine to +0.18 +- 0.10 %. Ratios above quote the independent-sample error used by the
threshold. The same muons in both arms also allow a paired error, and that one resolves an offset: fast vs refined
0.9927 +- 0.0019 nominal and 0.9934 +- 0.0011 moved, fast vs stock 0.9943 +- 0.0019 and 0.9953 +- 0.0011. The fast stop
fraction is 0.5-0.7 % low. Neither reference moves the first-foil deficit: 3,805 fast stops on the moved target, from
the saved fast arm used here (stop fraction 0.66612; the 0.66588 row above is an earlier arm).

Throughput: 14,260 muons/s on one A100 MIG slice, 42,270 on a whole A100, 184 on one CPU core; Geant4 83 muons/s on
one GPVM core.

## Device precision (pending change)

Every result above uses float64 on the device. A float32 device mode is validated on 100 CRV shards (within the
float64 seed-to-seed spread, pulse PE about -0.1 %) but is not on `main`.
