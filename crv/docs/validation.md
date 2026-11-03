# CRV validation

What the fast CRV chain has been checked against, on which sample, and with what result. Each row is one check;
"commit" is the `mu2e-crv-fastsim` commit the check ran on (or this repository's `main` on that date when only the
date is recorded). Thresholds were agreed before the comparisons were run (V0-V4: [`plan.md`](plan.md) section 5;
V-R1..V-R5: [`fastreco.md`](fastreco.md)). Open items and what is not validated at all are in
[`limitations.md`](limitations.md).

Samples: `g4ref_v1` is the Geant4 reference of the extracted (KPP) configuration (stock `ExtractedCRY`, run 1400,
20 shards of 50,000 generated events); `g4ref_v2` is the larger held-out reference (220 shards of 50,000). The tier-1b
library and its templates were built from production files and from odd `g4ref_v2` shards only, so the even `g4ref_v2`
shards are independent of every design choice.

## Current default chain (extracted_v04: `range_out` transport, tier 1b `strict_v5`)

The deck comparison runs the stock CRVDQM histograms (CrvStep, digi, pulse and cluster histograms of every sector
type) for two arms on identical CRY primaries and flags a histogram when its chi2/ndf (event-clustered errors, the
lowest bin of the energy and momentum histograms left out) exceeds max(2, 3 x the stock-vs-stock floor) (`crvfastsim.validation.compare`).
Arms: A0g = Geant4 steps + stock response and reco on the generator stream; A1 = fast steps + stock response and reco;
E2E = fast steps + fast response + fast reco; A2/A3/A4 = fast response / fast reco / both on Geant4 steps; A0' = A0
with reseeded digitisation (the floor).

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| A1 steps, `strict_v5`, paired per-shard statistic | A0g | `g4ref_v2` even shards 22-220, 5 M generated | cluster PE chi2/ndf; above 2000 PE fast / Geant4; histograms over paired 2; counts | cluster PE 1.09 (`strict_v4` 3.28); 0.991 +- 0.014 (0.923 +- 0.014); 31 / 147 (15), dev half 3 (6), confirmation half 20 (7), all occupancy and multiplicity; digis +0.33 +- 0.14 %, pulses +0.37 %, clusters +0.23 % | per-histogram limit | cluster PE closed; NPc, NDigis2 open (limitations); confirmation-half occupancy was tier-1b seed noise (row below) | 2026-11-02 | `eed5cdd` |
| E2E end to end, `strict_v5`, paired per-shard statistic | A0g | same, 5 M | cluster PE; above 2000 PE; histograms over paired 2 | 1.50 (`strict_v4` 3.26); 1.009 +- 0.014 (0.937 +- 0.014); 23 / 147 (21) | per-histogram limit | as A1 | 2026-11-02 | `eed5cdd` |
| V3 tier-1b steps, `strict_v5` | Geant4 steps | dev half 22-120, 2.5 M | NStep chi2/ndf; EX adjacent-layer correlation of descendant energy; descendant energy | 1.29 / 1.48 (two seeds; `strict_v4` 2.92); 0.43 +- 0.07 / 0.39 +- 0.02 (Geant4 0.40 +- 0.02, `strict_v4` 0.24); +0.40 +- 0.67 % | 10 % relative | pass | 2026-11-02 | `eed5cdd` |
| `strict_v4` reproduced | previous default tables | shards 22-40 | step and particle tables | identical | identical | pass | 2026-11-02 | `eed5cdd` |
| tier-1b seed, confirmation half | E2E, `strict_v4` and A0g | shards 122-220, 2.5 M | yields; per-SiPM occupancy; cluster PE | second seed vs `strict_v4`: digis +0.05 +- 0.15 %, pulses +0.03 +- 0.14 % (first seed +0.25, +0.26 %), per-SiPM chi2/ndf 0.18 (2.54); seed to seed digis -0.20 +- 0.17 %; vs A0g digis +0.28 +- 0.17 %, cluster PE 1.20 | per-histogram limit | the occupancy flags of the first seed are draw noise | 2026-11-02 | `179fed9` |

The rows below ran on the previous default `strict_v4` (hard-muon draws instead of root-coherent draws).

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| A1 steps | A0g | `g4ref_v2` even shards 22-220, 5 M generated | flagged histograms; clusters A1/A0g - 1 | 21 / 155 flagged; clusters +0.12 %, pulses +0.24 %, digis +0.19 % | 0 flagged | open (see limitations) | 2026-10-30 | `9fe268a` |
| E2E end to end | A0g | same, 5 M | flagged histograms; clusters | 13 / 133 flagged; clusters +0.10 %, pulses +0.21 %, digis +0.22 % | 0 flagged | open | 2026-10-30 | `9fe268a` |
| factorisation | A1 | same, 5 M | flagged histograms | 0 / 133; clusters -0.02 % | 0 flagged | pass | 2026-10-30 | `9fe268a` |
| A2 response | A0 (Geant4 steps) | `g4ref_v2`, 500 k | flagged histograms | 0 / 147; clusters +0.01 % | 0 flagged | pass | 2026-10-30 | `9fe268a` |
| A3 reco | A0 | same | flagged; pulse and cluster counts | 0 / 147; counts identical | 0 flagged | pass | 2026-10-30 | `9fe268a` |
| A4 response + reco | A0 | same | flagged histograms | 0 / 147 | 0 flagged | pass | 2026-10-30 | `9fe268a` |
| floor | A0 | same | flagged histograms | 0 / 177 | 0 flagged | pass | 2026-10-30 | `9fe268a` |
| A1 steps, paired per-shard statistic | A0g | same 100 shards, 5 M | flagged histograms (distinct quantities); cluster PE chi2/ndf (limit); cluster PE above 2000, fast / Geant4 | 11 / 155 (7): cluster PE, PEc, NPc, NStep, NDigis, digis per event; cluster PE 3.28 +- 0.40 (1.67), rms excess 3.8 %; 0.923 +- 0.014 | per-histogram limit, family-wise 1 % | open: cluster PE and multiplicities | 2026-11-01 | `9fe268a` |
| E2E end to end, paired per-shard statistic | A0g | same, 5 M | flagged histograms (distinct) | 12 / 133 (8), one of them a cluster-X bin edge at x = -3904 mm, not physics | per-histogram limit | open | 2026-11-01 | `9fe268a` |
| factorisation, paired per-shard statistic | A1 | same, 5 M | flagged histograms (distinct); cluster PE | 9 / 133 (5): ADC saturation bin, pulse PE (rms 0.41 %), cluster-PE tail +1 %, cluster-X bin edge; cluster PE 1.19 +- 0.32 (1.64) | per-histogram limit | small fast-response differences; cluster PE clean | 2026-11-01 | `9fe268a` |
| null of the paired statistic | A0 (A0' = reseeded digitisation) | 90 shards | flags; pull rms | 0 / 177; 1.013 +- 0.007 | 0 flagged | pass | 2026-11-01 | `9fe268a` |
| `range_out` transport | Geant4 primary-muon crossings | `g4ref_v2` even 22-220, 5 M | chi2/ndf of log10 visible energy below 200 MeV/c; of the stopping crossing's entry momentum | 5.42 -> 1.18; 16.66 -> 2.01 (earlier transport -> `range_out`); in-bar stops 715 G4 / 555 -> 643 | improvement, no histogram worse | adopted as default | 2026-10-30 | `9fe268a` |
| device transport (opt-in) | NumPy transport | 5 M generated | step yield; V1 | +0.04 +- 0.05 %; V1 pass | V1 | pass | 2026-10-30 | `a3a1612` |

The paired per-shard rows replace the three error rules of the deck comparison with one: the variance of the
per-shard difference of the two arms (both arms share the CRY primaries of each shard), a sign-flip reference of the
shard differences that keeps the correlations between bins, and one limit per histogram at a family-wise 1 % over a
comparison. The earlier time-histogram errors were 2.8-3.4x too large for a difference of arms that share the cosmic's
time; per-bar and per-channel occupancy flags of the earlier rule disappear. A flag no longer means a large
difference: the rms excess beside it gives the size. Same method as the tracker's
[`shard-statistics.md`](https://github.com/RobMina/mu2e-tracker-fastsim/blob/main/docs/shard-statistics.md).

## Validation levels V0-V4 (simulation)

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| V0 geometry | GDML export and `constructCRV` printout | extracted_v04, 576 bars | bar centre and half-length residual | max 3.6e-12 mm; same ids; no rotations | 1 um | pass | 2026-10-16 | `b73e869` |
| V1 crossings, stage C | Geant4 `g4ref_v1` | 20 shards, 1 M generated | crossings/event; path and visible-energy means and MPV; binned ratios | crossings EX -0.26 %, T1 +0.87 %, T2 -0.26 %, M -0.85 %; means within 0.43 %; MPV within 0.7 %; chi2/ndf 0.31-1.08 | 1 %; 0.5 %; 1 %; chi2/ndf <= 1.5 | pass | 2026-10-16 | `b73e869` |
| V1, extracted, reconfirmed with energy-ledger fixes | Geant4 | 1 M generated | as above, per sector group | pass | as above | pass | 2026-10-25 | `1041b86` |
| V1, Run 1A geometry | Geant4 Run 1A reference | 1 M generated | as above, grouped T1-T2 | pass | as above | pass | 2026-10-27 | `c3423f2` |
| V1, Run 2 geometry | Geant4 Run 2 reference | 1 M generated | as above | fails (below-plane scatter-in; diagnostics on `7ee7fa6`) | as above | **fail** | 2026-10-27 | `c3423f2` |
| V2 response on fast steps | full Geant4 chain (`ref`) | `g4ref_v1` held-out shards 11-20, 500 k generated | seq-0 PE median; PE spectrum; clusters per generated event; per-bar rate | PE median within 0.5 %; worst spectrum bin after 2 sigma 3.0 % (T1); clusters within 2.8 +- 3.3 % (M); per-bar chi2/ndf 0.44 | 2 %; 5 %; 1 % (2 sigma); 1.5 | pass (legacy library; superseded by A1 above) | 2026-10-19 | `59624f9` |
| V3 tier-1b steps, `strict_v2` templates | Geant4 steps | `g4ref_v2` shards 21-220, 10 M generated | descendant yield per sector group; step-histogram chi2/ndf | yields within 3.2 %; bar 41.5 -> 20.3, EX y 6.80 -> 2.91 | 10 % relative | pass on yields | 2026-10-28 | `892f053` |
| V3 tier-1b steps, chord placement | Geant4 steps | `g4ref_v2` even 22-220, 5 M | step-histogram chi2/ndf | all/y 3.23 -> 1.73, T1/y 4.53 -> 1.60, T2/y 5.42 -> 2.84; yields unchanged within errors | 10 % relative | pass on yields | 2026-10-29 | `ca63a05` |
| V4 photons (CPU) | Offline `MakeCrvPhotons` on the same steps | `g4ref_v1` shards 11-13, 24 M photons | photons per SiPM per event; arrival-time quantiles | largest +0.46 +- 0.26 % (M); times within 0.10 ns | 0.5 % (2 sigma); 0.2 ns | pass | 2026-10-19 | `59624f9` |
| V4 SiPM charges (CPU) | Offline `MakeCrvSiPMCharges` on the same photons | shard 11, 8 M photons, two seeds | PE per SiPM per event; time quantiles | largest -0.18 +- 0.15 % (T2); times within 0.17 ns | 0.5 % (2 sigma); 0.2 ns | pass | 2026-10-19 | `59624f9` |
| V4 fused device response | NumPy port | real G4 steps, shards 11-13, A100, PVC, B200, T4 | PE per SiPM; times | within 0.20 % and 0.14 ns (shard 11 T1 90 % time quantile is a known 0.2 ns noise case) | 0.5 %; 0.2 ns | pass | 2026-10-22 | `056e390` |

## Validation levels V-R1..V-R5 (reconstruction)

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| V-R1 waveforms | stock `CrvWaveformsGenerator` + `MakeCrvDigis`, same charges | MC, 22,095 events | digis, noise off; ADC KS, noise on | 565,579 of 565,579 digis sample-identical; noise width, shape and whiteness as stock | identical; KS p > 0.01 | pass | 2026-10-19 | `59624f9` |
| V-R2 pulses | stock `MakeCrvRecoPulses` on stock digis | MC 709,883 pulses; KPP data run 124155, 1.3 M pulses | finding; failed-fit flag; PE and time of good fits | MC identical, flag 99.996 %, 99.974 % within; data flag 99.979 %, 11 of 0.9 M waveforms differ in finding | identical; 99.5 %; 99 % within 0.5 % / 0.1 ns | pass (data: explained 11-waveform exception) | 2026-10-16 | `b73e869` |
| V-R3 coincidences | stock `CrvCoincidenceFinder` on stock pulses | 118,146 clusters, 40 shards (NumPy and JAX, CPU and GPU) | cluster lists | identical, including opposite-side attachment | identical | pass | 2026-10-26 | `1041b86` |
| V-R4 end to end | stock chain from the same CrvSteps | `g4ref_v1`, 20 shards, 50,008 events (NumPy and JAX) | PE median; binned PE; coincidences per event; EX layer efficiency | pass in every sector group | 2 %; 3 %; 1 %; 0.3 points | pass | 2026-10-21 | `df64880` |
| V-R5 objective | objective from the stock path | knob scan | position of the minimum | same minimum within the loop noise | <= 1/10 of the spread | pass | 2026-10-22 | `056e390` |

## Performance goals (agreed before the work started)

| Goal | Target | Result | Where | Date |
|---|---|---|---|---|
| P1 step generation | <= 2 ms per accepted event, one core | about 1.9 ms (shared host; to re-measure idle) | GPVM core, NumPy | 2026-10-20 |
| P2 photons + SiPM | >= 50x one core on one A100 | 51-57x (2,334-2,597 events with CRV steps per second) | one A100-80GB | 2026-10-22 |
| P3 end to end | >= 1 M accepted events per node-hour | 1.46 M | one Aurora node | 2026-10-21 |
| P4 response-only rerun | 1 M cached events <= 15 min | 10.4 min | one Polaris node | 2026-10-22 |
| P5 container | no CVMFS, < 5 GB, cold start <= 5 min | cold start 10.7 s | Polaris | 2026-10-21 |
| P6 reproducibility | bit-identical per seed; CPU and GPU statistically equal | met (float64): bit-identical across devices per seed | all endpoints | 2026-10-30 |

Throughput of the default chain per endpoint (cached CRY primaries to coincidence clusters, warm, events with CRV
steps per second, transport and tier 1b on the device, float64): one GPVM core 55; EAF A100 MIG 1g.10gb slice 163;
Polaris node (4 A100) 990; Aurora node (12 PVC tiles) 1,212; UVA 8 x A100 1,695. Stock Offline on one grid job: 0.92.

Production throughput at the current defaults (2026-11-02, umbrella `main` `afea868`: device float32, XLA autotuning
off; `runchain` over 11 held-out shards of 50,000 generated events, persistent compilation cache warm, steady state
over shards 2-11, generated events per second; in brackets the number of stock single-core grid jobs replaced, at
21.2 generated events/s per grid job, Geant4 + stock response and reco on the same events):

| Device | Generated events/s | Events with CRV steps/s | Grid jobs replaced |
|---|---:|---:|---:|
| RTX PRO 6000 | 18,300 | 820 | 862 |
| RTX PRO 6000 MIG 1g.24gb slice | 16,790 | 750 | 791 |
| A40 | 12,270 | 550 | 578 |
| A100-SXM4-80GB | 10,710 | 480 | 504 |
| V100 | 7,930 | 350 | 373 |
| one CPU core, NumPy chain (one pass of 10 shards, two seeds) | 860 | 39 | 40 |

Without a warm cache the first shard takes 100-180 s (compilation) and later shards 1.5-2x their steady time. One job
per device; jobs on one card have differed by up to 30 %.

## Speed switches

Measured 2026-11-01 on the default chain: ten held-out shards 22-40 (500,000 generated events), two seeds, NumPy transport
and tier 1b, one A100-80GB node. Identity is byte equality of every retained array per seed: pulses, clusters, members,
digis, charges. Speedups are warm-pass ratios; where one prepared process could hold both arms they are paired over
alternating passes (`scripts/switch_ab.py`), otherwise separate processes in the order A B B A.

After the switches became defaults the chain was re-checked against the code with the old defaults, one process each,
same shards: NumPy chain byte-identical at both seeds (reco 190 -> 116 s and 189 -> 105 s per pass); GPU chain at float32
on an A100-80GB byte-identical at seed 20261027.

| Switch | Chain, precision | Identity | Warm speedup | Decision |
|---|---|---|---|---|
| `CRVFS_CPU_ATTACHMENT` | NumPy | identical | 1.154x (1.148-1.161) | on by default |
| `CRVFS_CPU_FIT_NORMAL_CACHE` | NumPy | identical | 1.028x (1.026-1.035) | on by default |
| both | GPU, float32 (NumPy redo path) | identical | 1.015x (1.013-1.018) | |
| `CRVFS_CPU_FIT_WORKSPACE` | NumPy | identical | 1.007x (1.004-1.013); none on top of the cache | `park/cpu-speed` |
| `CRVFS_CONTEXT_HASH_WORKERS` | GPU, float32 | identical | 1.042x at 4, in `e2e-bench --reuse-context` only | default 4 |
| `CRVFS_PREPARED_LIBRARY_CACHE` | GPU, float32 | identical | none warm (0.984x, overlapping); preparation 52.5 -> 19.9 s, first creation 70.9 s | kept: a site path |
| `CRVFASTSIM_COMPACT_DISPATCH` | GPU, float32 and float64 | **not identical** (cluster PE, time, position, slope only; float32 `cl_pos` up to 0.07 mm, float64 5e-9 mm) | 1.05-1.11x warm; cold pass 1.2x slower (more shapes to compile) | `park/cpu-speed` |

## Physical readout mask

The mask of physically present SiPMs (Offline `hasCMB`, [readout-mask.md](readout-mask.md)) is applied by default
since 2026-11-03. On the extracted configuration it changes nothing: every single-sided sector has its reflector on the
missing side, so no photon reaches an absent SiPM (also Run 1A; `tests/test_readout_default.py`). Checked against the
chain without the mask on three held-out shards (22, 24, 26; strict_v5, seed 22, e2e-bench output fingerprints of
pulses, clusters, members, digis and charges): byte-identical on the NumPy chain (CPU), the JAX chain and the fused
device response (MIG 1g.24gb, float32).

## Device precision

The device kernels run in float32 by default since 2026-11-01 (`MU2EFS_DEVICE_FLOAT`; float64 one variable away).
Validated on 100 shards on V100 and A100 MIG slices: every aggregate within the float64 seed-to-seed spread; seq-0
pulse PE float32 - float64 between -0.03 and -0.14 %, bounded at <= 0.3 % (2 sigma) per sector group; the device
tier-1b sampler passes the deck rule at float32 (0 of 21 shapes). Rows above that say float64 were measured at
float64. Run-to-run identity on the GPU (2026-11-01, held-out shards 22 and 24, every stage hashed per column): with XLA
autotuning off (the default `MU2EFS_DEVICE_DETERMINISTIC`), separate processes byte-identical at float32 on the RTX PRO
6000 MIG slice, the full RTX PRO 6000, the A100 and the V100, and at float64 on the MIG slice; with autotuning on they
were not on the first three (the V100 happened to repeat). Cost +0.4-0.8 % warm, cold pass 16-28 % faster. Details:
[device-precision.md](device-precision.md#reproducibility).
