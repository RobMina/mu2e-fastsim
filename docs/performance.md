# Performance and validity scorecard

Every chain is reported on three questions: (1) does it reproduce the stock Geant4 + Offline result by the agreed
metrics and the stock DQM/validation histograms, (2) how fast is it compared with stock on a grid node, and (3) what
does an event cost on each kind of GPU. This page answers (1) and (2) with summary statistics; the full validation
tables are in each repository's `docs/validation.md` ([CRV](https://github.com/RobMina/mu2e-crv-fastsim/blob/main/docs/validation.md),
[tracker](https://github.com/RobMina/mu2e-tracker-fastsim/blob/main/docs/validation.md),
[core](https://github.com/RobMina/mu2e-fastsim-core/blob/main/docs/validation.md), beam in [validation.md](validation.md)).
Allocation charges depend on the site's rates and are not given here.

## How the comparison is made

- **Event unit.** Cosmic chains count *generated* CRY events, because the fast and the stock chain then process the
  same event universe (about 4.4 % of generated events leave steps in the CRV, 4.9 % pass the stock `PrimaryFilter`).
  The beam counts muons entering the detector solenoid; the conversion-electron (CE) chain counts electrons.
- **Stock reference.** The stock chain for the same events and the same stage boundary, timed on a Fermilab grid
  node, one job per core:
  - cosmic CRV: Geant4 (`ExtractedCRY`, SimJob MDC2025ay) plus stock CRV response, digitisation, pulses and
    coincidences, measured in one grid job of the extracted reference (50,000 generated events: Geant4 2,287 s, CRV
    response and reco 68 s): **47.1 ms per generated event, 21.2 generated events/s per grid slot**;
  - cosmic tracker: the same Geant4 plus stock digitisation (49.9 ms per accepted event) and KKLine reco (about
    19 ms per accepted event): **49.1 ms per generated event, 20.4 per second per grid slot**;
  - beam: the stock muon-beam stage of a Mu2eBO design-point evaluation on the grid, for the 49-foil target designs
    of the stopping study: 21.6 ms of CPU per event (fifteen jobs of 200,000 events per design point, from the job
    logs). The stock stage starts at the TS3 plane and the fast one at the detector-solenoid entrance; 0.1615 +- 0.0012
    muons per stock event reach that entrance, so per DS-entering muon (equivalently per target stop) the stock cost is
    134 ms: **7.5 muons/s per slot**. The fast rate it is compared with is measured on one of those 49-foil designs;
  - CE: the stock `CeEndpoint` production stage (SimJob MDC2025ba) on grid nodes, 40.9-42.7 ms per generated
    electron over three jobs (Geant4 18-19 ms of it): **24 electrons/s per slot**.
  Geant4 is 93-97 % of the cosmic stock time, and 93 % of it is spent on cosmics that never reach a detector.
  The grid node of the reference job is about 1.9x slower than an interactive Mu2e GPVM core on the same events
  (the GPVM rerun of the same job reproduces its accepted events exactly in 24.4 ms of Geant4 per generated event);
  across the 20 jobs of the reference sample the grid wall time per job ranged over a factor of two. Per GPVM core,
  every multiplier below is about half.
- **Multiplier of grid jobs** = fast events per second on one device / stock events per second per grid slot: the
  number of concurrent single-core stock grid jobs that one device replaces at the same event rate. It counts payload
  only; grid queueing, setup and stragglers, which typically make a grid stage take 3x its payload, are not included.
- **Fast rates** are steady-state production rates (compiled kernels, persistent compilation cache warm, inputs on
  local storage), cached CRY primaries to coincidence clusters (CRV) or to line fits (tracker), at the repository
  defaults: device kernels float32 for the CRV, float64 for tracker and beam; XLA autotuning off. CPU rates are
  single NumPy (or JAX-on-CPU) processes on one core.

## Scorecard

| Chain | Validity (summary) | Fast rate | Grid slots replaced |
|---|---|---|---|
| CRV default chain (`range_out` transport, tier 1b `strict_v5`: root-coherent secondary draws) | V0, V1, V4, V-R1..V-R5 pass. 5 M held-out events, paired per-shard statistic: fast steps + stock response vs Geant4 + stock, cluster PE chi2/ndf 1.09 (3.28 with the previous tier 1b), end to end 1.50; clusters above 2000 PE 0.991 +- 0.014 (was 0.923); adjacent-layer correlation of secondaries 0.39-0.43 (Geant4 0.40); counts within 0.4 %. Open: clusters per event (NPc), wide-range multiplicities (events with more than 100 digis 0.83 +- 0.05 of Geant4), secondary steps at the bar ends; one tier-1b seed moves the yields by about 0.2 %. Float32 device output within the float64 seed spread (previous tier 1b) | one CPU core 860 generated/s; one GPU 7,900-18,300 generated/s (table below) | 40 per core; 370-860 per GPU; 1,100-1,360 per HPC node (older benchmark) |
| tracker, extracted configuration | T-V0, T-V1 primary-muon steps pass; fast digis vs stock on the same steps 0 of 19 flagged; hits, clusters and seeds identical to stock; device chain vs NumPy 0 of 11 flagged. Open: T-V3 track yield -1.42 +- 0.37 % (default), -0.85 +- 0.37 % (opt-in `--scatter-in`); muon rate -0.82 +- 0.26 %; secondary placement; F1 fit 94.2 % vs 99.5 % | one CPU core 3,030 generated/s; 40-core node 111,500; one RTX PRO 6000 MIG 1g.24gb slice 22,160; one A100 14,270 | 149 per core; 5,480 per 40-core node; 1,090 per MIG slice; 700 per A100 |
| beam stop fraction | fast / Geant4 on the same muons 0.9976 +- 0.0058 (nominal target), 0.9950 +- 0.0031 (49 foils); finer Geant4 steps do not move it (1.0016 +- 0.0020); with paired errors fast is 0.5-0.7 % low | one GPU 10,100-45,900 muons/s on the nominal target (table below); 9,640 (MIG slice) and 40,900 (A100-40GB) on a 49-foil design | 1,290 per MIG slice, 5,470 per A100 (49-foil design) |
| Mu2eBO stopping stand-in (`beam_point` on each design point's target) | 65 Geant4-evaluated design points, no fitted scale (muons per event at the stopping-target region predicted from a Geant4 pass on the boards' own beam sample): Geant4 / fast 1.0045 +- 0.0005 (+- 0.0077 from the predicted scale), rms 0.42 % (0.73 % statistical, an upper bound: the points share the fast muons), chi2/ndf 0.76, rank correlation 0.998 +- 0.001 | as beam; one design point (105,350 muons) 3-5 min through the kit including compilation | as beam; per design point the stopping objective drops from about 4 h (grid chain) to 4-5 min |
| Mu2eBO sob stand-in (`beam_sob`: stops, CE chain, calorimeter proxy, the studies' approximate sensitivity) | 65 Geant4-evaluated design points, no fitted scale: Geant4 / fast 0.9920 +- 0.0006 (+- 0.0077 from the stops normalisation), rms 0.46 %, chi2/ndf 0.40, rank correlation 0.993 +- 0.003; the 0.8 % offset is the calorimeter proxy (+1.7 %) against the stops (-0.45 %) and the option-0 design points (-0.4 %) | one design point (100,000 CE) through the kit 20.5 min from launch to result row: queue 3 min, job 16.7 min on 32 CPU cores (stops 76 s, CE chain and calorimeter 858 s, 117 electrons/s) | per design point the sob objective drops from about 4 h (grid chain, critical path) to about 20 min |
| electrons, conversion-electron chain to the tracker entry (opt-in) | slab physics vs Geant4 pass (loss chi2/ndf 0.5-1.4; 1 X0 W stopped 18.41 vs 18.29 +- 0.09 %; reflected 0.347 +- 0.013 vs 0.336 % with the opt-in form-factor threshold). CE chain vs Geant4 CeEndpoint at the tracker entry, inside a selection that implies the production filter: yield per generated electron 1.004 +- 0.003; momentum, angle, radius and time-of-flight chi2/ndf 0.8-1.3; E - E_CE MPV -0.37 vs -0.35 +- 0.02 MeV, FWHM 0.675 vs 0.677 +- 0.024 MeV, 0.25 MeV tail bins to -2 MeV within 1-2 sigma Device chain (`--backend device`, one JAX loop) vs the NumPy chain on the same starts (400k target stops and the 108k Geant4 starts, r_max > 450): yield per start 0.9999 +- 0.0007 / 1.0005 +- 0.0010 (float64), 1.0010 +- 0.0007 / 1.0000 +- 0.0010 (float32); E - E_CE 0.25 MeV bins from -2 to 0 MeV within 2.4 sigma (paired); same-start momentum +0.001 +- 0.008 MeV/c (float64), +0.012 +- 0.008 MeV/c (float32); against Geant4 the device yield is 1.0043 +- 0.0028 (float64) / 1.0054 +- 0.0028 (float32) | device chain, one RTX PRO 6000 MIG 1g.24gb slice with 4 cores: float32 about 70,000 electrons/s steady (whole process with 1.6 M electrons: 24,900/s, fixed cost about 17 s plus a 10 s tail of slow electrons), float64 8,600/s (process with 400k: 3,900/s); one CPU core: float32 170-240/s (process with 40k: 116-135/s), float64 106-110/s. NumPy chain with JAX layers: 37/s per core, 132/s per MIG slice. A stock event (24 electrons/s per grid slot) simulates every particle in the whole detector; the fast chain carries the primary electron to the tracker entry | device: about 2,900 per MIG slice (float32), 360 (float64); 7-10 per CPU core (float32), 4.4-4.6 (float64) |
| tracker with the DS field (opt-in, in development) | first passage of Geant4 tracker-entry states, field-on chord transport: straws per track +0.017 +- 0.070 (muons), +0.055 +- 0.019 (CE); ionisation mean +0.29 +- 0.18 %; momentum above 1 GeV/c biased (no radiative losses yet) | one CPU core 6 muons/s, 4 CE/s (transport only) | stock field-on reference not yet timed |

## Trigger stock basis (replay diagnostic)

R7 is paused as of 2026-11-05 06:10 EST; no R7 jobs are running. The first C parent
completed, but its roughly 10 min 32 s runtime is a bounded compatibility observation,
not a response/trigger acceleration benchmark. The other nine parents and full C
trigger are unlaunched. See [pause/resume scope](../trigger/PAUSE_RESUME.md).

Finished Run 1A production digitization logs give trigger-module wall work of **8.17-8.32 ms per processed CE dts
event** and **9.55-10.24 ms per processed cosmic dts event** over two jobs each. Shared labels are counted by
`sum(average * invocations)` across first-execution paths, excluding original digitization, MC truth decoration,
source and output. Full-event wall means including those other costs are 29.88 ms (CE) and 52.65-53.32 ms (cosmic).
The separately recovered production job matching the complete-history CE control has 28,044 processed events:
8.12998 ms trigger work per input event, 29.9453 ms full-event wall mean, 886.946 s total CPU and 957.627 s real.
The complete-history replay's verbose validation run uses 545.344 s CPU / 562.661 s real; PrintModule I/O and
startup/end costs are included, so it is not a quiet timing comparison or a fast-trigger throughput claim.
These ranges are observations, not uncertainty estimates or isolated trigger CPU latency; input dts events differ
from generated events and trigger/truth-selected stored digis. The five CE LoopHelix fits account for about 71%
of trigger work; the cosmic robust multi-helix finder alone accounts for 28-29%. These stock-production
attributions alone supply no fast-trigger rate or grid-slot multiplier; the measured hybrid quiet pair below
also does not establish an end-to-end fast-chain multiplier.

A complete stored-history quiet comparison processes 28,044 events in each arm, stock then tracker-fast with
stock calorimeter, on one CPU allocation. Both arms exactly preserve their respective saved verbose-baseline
original and replay packed decisions. Attributed trigger wall work is **12.986762 → 12.427416 ms/event (-4.307%)**;
full-event wall mean **15.2450 → 14.7262 ms (-3.403%)**. However, total art real time is
**559.837428 → 563.486789 s (+0.652%)**, and cold-cache launcher elapsed also increases. Thus **no whole-job
acceleration is measured**. This is one fixed-order pair, not randomized replication or a timing-significance
bound; source layouts, fit workload/acceptance and fresh per-invocation CVMFS caches differ. The production
log hardware/population/digitization basis prevents a direct production speedup ratio. It is not full fast-chain,
JAX or SU/event timing. Five shared fits account for 71.17% stock / 70.38% hybrid trigger work; the two BO-final
selections together account for 0.1523% / 0.1597%. Per-path module attribution is not independent path latency,
and full-event maximum/median are not isolated trigger p99.

A quiet same-stock CE replay of 21,078 stored selected digi events measures 11.22 ms per event in the event loop
(89.1 stored events/s), of which 9.54 ms is trigger-module work. Total art real time is 309.62 s (68.1 stored
events/s including startup/end). These are a selected denominator on a different CPU, not a speedup relative to
the production dts basis or a fast-digi result. [Replay and timing tools](../trigger/README.md).

## Measured BO final-selection CPU batch

Both unchanged measured TPR final selections together evaluate all 28,044 stored events (18,038 tracks,
padding width one, explicit zero/absent collections) in **0.120646 ms median warm batch time** over 30 repeats
after three warmups on an arm64 laptop CPU with JAX 0.11.2/x64. Observed batch range is 0.107250–0.139208 ms,
batch p95 0.133913 ms. All outputs are synchronized. Compilation is separate: lowering 28.365 ms, compile
65.507 ms, first execution 0.321 ms, device placement 0.849 ms. Input validation 18.500 s and additional
load/padding/target checks 9.302 s are excluded from warm time. JAX caches are cleared after validation;
persistent compilation caching is disabled for compile, but backend initialization is already paid.

This is a fixed-reconstruction selection microbenchmark, with actual upstream fast reach supplied rather than
emulated, and composed outputs checked against measured targets. The CPU process is not core-pinned or
thread-limited; no single-core rate follows. The amortized 0.004302 microseconds/input-event is **not** an
individual-event latency. There is no same-hardware C++ comparison, full-trigger acceleration, GPU or SU claim.
These cuts account for about 0.15% of measured quiet trigger work; dominant reconstruction remains unported.
See [selection details](../trigger/JAX_SELECTION.md).

## Per GPU: CRV default chain and beam stop fraction

One GPU per job, the defaults of `main` (2026-11-02). CRV: production `runchain` over 11 shards of 50,000 generated
events, persistent compilation cache warm, steady state over shards 2-11 (in brackets the whole 11-shard job including
setup and the first shard); grid slots at 21.2 generated events/s. Beam: `beam_point` on the nominal 37-foil target,
105,350 DS-entering muons, warm passes (a 49-foil design runs 4-6 % slower); no stock time per event exists for the
nominal target on the grid at the same boundary, so the beam column gives rates only.

The CRV rates were measured with the previous tier-1b set (`strict_v4`). The default `strict_v5` samples tier 1b at the
same CPU rate (16.5 k against 16.3 k generated events/s on one core; the whole CPU chain is unchanged within its repeat
spread) and adds about 13 s of one-time setup per process; the GPU rates have not been re-measured.

| GPU | CRV generated events/s | CRV grid slots | Beam muons/s (nominal target) |
|---|---:|---:|---:|
| RTX PRO 6000 (full card) | 18,300 (7,210) | 862 | 31,560 |
| RTX PRO 6000 MIG 1g.24gb slice | 16,790 (7,410) | 791 | 10,090 |
| A40 | 12,270 (5,020) | 578 | 10,790 |
| A100-SXM4-80GB | 10,710 (3,920) | 504 | 45,850 |
| V100 | 7,930 (3,420) | 373 | 22,970 |

The CRV device kernels are float32 by default and the beam's float64, which is why the beam ranks the cards with
full-rate FP64 (A100, V100) first and the CRV those with high FP32 throughput. Without a warm compilation cache the
first shard takes 100-180 s and later shards 1.5-2x their steady time while new shapes compile. Other cards, from a
benchmark that repeats one shard (it includes a per-pass integrity audit, so it runs at about half the production
rate): B200 5,780 generated events/s, A100-40GB 4,600, A6000 4,120, RTX 3090 2,720, RTX 2080 Ti 2,110. Whole nodes
from the same benchmark: ALCF Polaris (4 A100-40GB) 1,100 grid slots, ALCF Aurora (12 PVC tiles) 1,360. One job per
card was measured; jobs on the same card have differed by up to 30 %, so rankings closer than that are not established.

## Limits of these numbers

- Payload rates. A production job also pays compilation (one to a few minutes per process without a persistent
  compilation cache, seconds with one), library loading and input streaming; a short job is dominated by them.
- The stock reference is one grid job per cosmic chain; grid nodes differ by up to 2x.
- The tracker and CRV chains share the Geant4 stage in stock: a fast job that runs both on the same primaries
  replaces one stock job, not two.
- The CE and field-on tracker chains are under development: their rates are first CPU measurements.
