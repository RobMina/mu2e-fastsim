# Validation

Summary of what each stage is validated against. The full tables (sample, metric, result, threshold, verdict, date,
commit) are in each repository's `docs/validation.md`: [CRV](https://github.com/RobMina/mu2e-crv-fastsim/blob/main/docs/validation.md),
[tracker](https://github.com/RobMina/mu2e-tracker-fastsim/blob/main/docs/validation.md),
[core](https://github.com/RobMina/mu2e-fastsim-core/blob/main/docs/validation.md); the beam stage is below. Thresholds
were agreed before the comparisons were run. What is open or not validated: [limitations.md](limitations.md). Throughput against stock and a one-table summary per chain: [performance.md](performance.md).

## Summary

| Stage | Check | Result | Verdict |
|---|---|---|---|
| CRV | V0 geometry; V1 muon crossings vs Geant4 (extracted, Run 1A) | 576 bars to 3.6e-12 mm; crossings within 0.9 % per sector group | pass |
| CRV | V1 Run 2 geometry | below-plane scatter-in missing | **fail** |
| CRV | V4 photons and SiPM charges vs stock (CPU and GPU) | within 0.5 % and 0.2 ns | pass |
| CRV | V-R1..V-R5 waveforms, pulses, coincidences, end to end, objective | identical or within thresholds | pass |
| CRV | default chain vs Geant4 + stock, 5 M held-out events, paired per-shard statistic (family-wise 1 % limit per histogram) | 11/155 histograms flagged with fast steps (7 quantities: cluster PE, multiplicities), 12/133 end to end, 9/133 fast vs stock response+reco on the same steps (small: ADC saturation bin, pulse PE 0.4 % rms); null 0/177; counts within 0.25 % | open |
| tracker | T-V0 geometry; T-V1 primary-muon steps vs Geant4 | 0 steps outside their straw; every T-V1 threshold | pass |
| tracker | digis, hits, clusters, seeds vs stock on the same input | within the reseed floor; identical | pass |
| tracker | F1 line fit vs stock KKLine | acceptance agreement 94.2 % default, 95.6 % opt-in time model (target 99.5 %) | not met, frozen |
| tracker | T-V3 reco yield on fast steps vs Geant4 + stock | -1.42 +- 0.37 % default, -0.85 +- 0.37 % opt-in `--scatter-in` (threshold 1 %) | **fail** (default), pass on central value (opt-in) |
| tracker | any-primary-muon rate | -0.82 +- 0.26 % default; +0.14 +- 0.26 % with opt-in `--scatter-in` | open (default), pass (opt-in) |
| beam | stop fraction, nominal target, vs Geant4 on the same muons | ratio 0.9976 +- 0.0058 | pass |
| beam | stop fraction, 49-foil geometry move | ratio 0.9950 +- 0.0031 | pass |
| beam | stop fraction under much finer Geant4 steps, same muons | refined / stock 1.0016 +- 0.0020 nominal, 1.0019 +- 0.0011 moved | pass |
| beam | stop fraction vs refined-step Geant4 | ratio 0.9927 +- 0.0058 nominal; 0.9934 +- 0.0031 moved (2.15 sigma) | pass nominal, **fail** moved |
| core | JAX loss sampler vs NumPy; device CRY vs stock CRY (5 M events each) | within criteria; accepted rate 1.00014 +- 0.00063, no spectrum flag in 2,534 bins | pass |
| CRV | the chain on device-CRY primaries vs the same chain on the stock gen tables, 5 M events each, per live second | digis 1.0006 +- 0.0033, pulses 1.0004 +- 0.0033, clusters 0.9989 +- 0.0030; cluster PE shape chi2/ndf 0.98 | pass |
| CRV | device transport + device tier 1b (`strict_v5`, float32) vs the host chain, 5 M held-out events, paired per-shard statistic | 0 of 133 histograms flagged end to end, 0 of 155 with stock response | pass |
| electrons | dE/dx vs ESTAR | radiative 0.6 %, collision 1 % | pass |
| electrons | 1 X0 tungsten, 105 MeV e-, vs Geant4 | loss chi2/ndf 1.0-1.3; exit angle 1.0-1.3; stopped 18.28 vs 18.41 +- 0.09 %; reflected 0.336 vs 0.347 +- 0.013 % (form factor only above 100 MeV, the core default) | pass |
| electrons | field stepping (`ehelix`) vs analytic helices and RK4; no Geant4 comparison | helix 1e-9 mm; 18 um median over 3.7 m of the DS map against 1 mm RK4 | pass |
| electrons | device stepper `eslab_jax` vs NumPy `eslab`, no field | loss chi2/ndf 0.58-1.30, exit angle 0.60-1.21 | pass |
| conversion electrons (`cechain`, compared with `cecompare`), stopping target to the tracker entry plane (VD TT_FrontHollow, z = 8539.9 mm) in the Mau13 DS map | Geant4 CeEndpoint dts (MDC2025ac, option 4), 108,326 kept of 200,000 generated; fast on the same starts and on 400k from a separate target-stop sample | inside p > 60 MeV/c, helix r_max > 450 mm (implies the production filter): yield per generated 1.004 +- 0.003 of Geant4 (r_max > 500 / 550: 1.007 +- 0.004 / 1.008 +- 0.006); chi2/ndf (39) momentum 0.89, cos theta 1.19, r 1.34, r_max 0.82, time of flight 0.84; same-start time of flight -0.001 +- 0.004 ns, momentum +0.024 +- 0.011 MeV/c; E - E_CE MPV -0.37 +- 0.02 vs -0.35 +- 0.02 MeV, FWHM 0.675 +- 0.009 vs 0.677 +- 0.024 MeV, fractions in 0.25 MeV bins from -2 to 0 MeV and below -2 MeV within 1-2 sigma (0.0002-0.0007) | pass |
| conversion electrons, device chain (`cechain --backend device`, `cechain_jax`: one JAX loop with every layer type in one slab kernel), float64 and float32, MIG slice | the NumPy chain on the same starts (cecompare `--mode pair`, paired errors): 400k isotropic CE from target stops and the 108,326 Geant4 starts; and Geant4 as above | r_max > 450: yield per start 0.9999 +- 0.0007 (stops) / 1.0005 +- 0.0010 (Geant4 starts) float64, 1.0010 +- 0.0007 / 1.0000 +- 0.0010 float32; E - E_CE 0.25 MeV bins from -2 to 0 MeV and below -2 MeV: paired pulls within 1.6 sigma (float64), 2.4 sigma (float32); chi2/ndf (39) momentum, angle, radius, r_max, time of flight 0.3-1.0; same-start momentum +0.001 +- 0.008 / -0.001 +- 0.011 MeV/c (float64), +0.012 +- 0.008 / +0.006 +- 0.011 MeV/c (float32), time of flight within 0.003 ns; against Geant4 (stops) yield 1.0043 +- 0.0028 (float64), 1.0054 +- 0.0028 (float32), tail bins within 2.4 sigma, as the NumPy chain | pass |
| sob stand-in (`cesob`): calorimeter-acceptance proxy (helix path >= 17.8 mm in the Offline crystal map; the disks' inner Al and carbon-fibre rings crossed with the slab physics) | Geant4 visible calorimeter energy > 10 MeV (EdepAna) of the same 108,326 MDC2025ac CeEndpoint electrons, the proxy run on their Geant4 tracker-entry states | selected per electron at the entry: 0.9559 vs 0.9407 (+1.6 % relative; without the inner rings +3.5 %); electrons reaching only the second disk 0.961 vs 0.902. Residual: electrons that graze the first disk's inner edge and cross its back plate and front-end electronics (not modelled) | stated residual |
| sob stand-in (`cesob`): the design studies' approximate CE sensitivity | the studies' own estimate on 66 Geant4 EdepAna ntuples | the port reproduces the studies' sob on all 66 (relative difference 0) | pass |
| sob stand-in (`cesob`) on 65 Geant4-evaluated design points of the stopping-target study (stops on the study's beam sample with no fitted scale, 200,000 CE per point) | the points' full-simulation sob (FTFP_BERT, EM option 0) | Geant4 / fast 0.9920 +- 0.0006 (+- 0.0077 common, from the stops normalisation), rms 0.46 % (0.77 % expected), chi2/ndf 0.40, rank correlation 0.993 +- 0.003; factors: stops per event 1.0045, CE acceptance 0.9832 (the proxy residual), signal per stop in the window over acceptance 1.004 (same window on 60 of 65 points). Option 0 vs the option-4 fast chain: the design points' selected electrons lose less (fractions more than 1 / 2 MeV below the birth energy at the tracker entry 0.8 +- 0.1 % / 1.4 +- 0.1 % lower, relative), worth +0.4 % on sob; not corrected | pass (offset stated) |

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

## Device precision

The CRV device kernels run in float32 by default since 2026-11-01 (`MU2EFS_DEVICE_FLOAT`), validated on 100 CRV shards:
within the float64 seed-to-seed spread, pulse PE 0.03-0.14 % lower, bounded at 0.3 %
([crv `docs/device-precision.md`](https://github.com/RobMina/mu2e-crv-fastsim/blob/main/docs/device-precision.md)).
The beam results above are float64, as beam and tracker JAX code stay.
