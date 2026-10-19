# mu2e-crv-fastsim: a Geant4-free, GPU-accelerated simulation of the KPP (extracted) CRV cosmic sample

Status: **agreed 2026-10-15.** Sections 4 and 5 (performance goals, validation thresholds) are agreed as written,
and so is a Geant4-built library for the non-muon component (tier 1b). Implementation starts after ALCF access.

## 1. Goal and scope

Replace the Geant4 stage of the Mu2e extracted-configuration cosmic simulation (no magnetic field, no beam) with a fast
simulation that runs in a container, uses GPUs on ALCF Polaris, and is submitted through the ALCF IRI API. It serves
the CRV MC/data agreement work: response tuning (light yield, timing, thresholds), per-module alignment closure with
moved geometry, and samples comparable in size to the KPP cosmic data sets.

- **Stage 1 (this plan): the CRV.** Output is the same `CrvStep` collection the Geant4 chain writes, so the stock Offline
  CRV digitisation, reco and EventNtuple run unchanged on it. A GPU port of the CRV response follows.
- **Stage 2 (later): the tracker** (`StrawGasStep`s), same pattern.
- Not in scope: on-spill, field-on or full-CRV configurations; the calorimeter.

## 2. Target resources

| | ALCF Polaris | ALCF Aurora |
|---|---|---|
| GPUs per node | 4x NVIDIA A100 40 GB (NVLink), 1x EPYC Milan 32c, 512 GB | 6x Intel Data Center GPU Max (PVC), 2x Xeon Max 104c |
| Programming model | CUDA; JAX and PyTorch native | SYCL/oneAPI; PyTorch via Intel XPU, JAX via the Intel OpenXLA plugin (unverified) |
| IRI API job submission | **yes** (resource `55c1c993-1124-47f9-b823-514ba3849a9a`, PBS behind it) | **no**: not implemented in the ALCF IRI API |
| Hackathon access (1 month) | project `GenesisHackathonOct26`, reservation queue `R7728155` | reservation queue `R8914313` |
| CVMFS | none | none |

**Decision: Polaris is the primary target, driven entirely through IRI.** Aurora is a portability stretch for the GPU
kernels only, and needs direct PBS submission. Celeritas and Opticks have no Intel backend, which is one more reason
for a JAX/PyTorch design.

## 3. Baseline: what the Geant4 chain costs and produces (measured 2026-10-15)

### 3.1 Setup

| Item | Value |
|---|---|
| Host | a Mu2e interactive machine (AMD EPYC VM, idle at start), one job |
| Software | Musing `SimJob MDC2025ay` (Offline v13_39_00, p107, Geant4 11.3.2) |
| Sim | stock `ExtractedCRY.fcl`, run 1400, 40,000 generated events |
| Digi | stock `digitize/Extracted.fcl` with `Sim_best v1_5`, every accepted event written |
| Tools and raw outputs | job configs in `profile/`; summaries by `crvfastsim g4cost` / `census` (`crvfastsim/README.md`); outputs `$CRVFS_DATA/profile/`; frozen summaries in `profile/results/{timing_summary,census}.json` |

### 3.2 Cost

| Quantity | Value |
|---|---|
| Accepted by `PrimaryFilter` (trk OR calo OR CRV) | **1,947 / 40,000 = 4.87 %** (production MDC2025as: 24,806 / 500,000 = 4.96 %, consistent) |
| `g4run` time, all events | 844 s = 21.1 ms per generated event = **0.433 s per accepted event** |
| `g4run` time spent on events that are then rejected | 786 s = **93 %** of all G4 time |
| `g4run` time on accepted events only | mean 29.3 ms, median 18.2 ms, p99 152 ms |
| CRY generator | 0.50 ms per generated event |
| CPU time, whole sim job | 920 s user (wall 1,570 s, slowed by per-event TimeTracker/MemoryTracker SQLite writes on CephFS) |
| CRV response per accepted event (`CrvPhotons` + `CrvSiPMCharges` + `CrvWaveforms` + `CrvDigi`) | mean **19.8 ms**, median 15.9 ms, p99 72 ms; `CrvPhotons` alone 18.2 ms (92 %) |
| Full digitisation per accepted event (adds tracker, calo, trigger) | mean 49.9 ms |
| Memory | sim VmHWM 2.06 GB; digi VmHWM 0.74 GB |
| Output size | dts 7.2 kB per accepted event; dig 9.0 kB per accepted event |

Three readings:

1. **Most of the Geant4 cost buys nothing.** 93 % of G4 time goes to the 95 % of cosmics that never reach a detector.
   A geometric pre-selection alone, ray against detector boxes before transport, would cut the per-accepted-event
   cost from 0.43 s to about 0.03 s. This is the same generator-level filter the datasets page lists as an open
   question.
2. **On accepted events, G4 and the CRV response cost about the same** (29 ms against 20 ms). So a Geant4-free step
   generator only pays off together with a fast response. That is the case for the GPU work.
3. **Scale.** The MDC2025as production generated about 1.0e10 events, about **59 k core-hours** of `g4run` at this
   rate.

### 3.3 What reaches the CRV

`crvfastsim census` splits the CRV visible energy of the accepted events by origin: the primary muon itself, its
descendants (delta rays, decay electrons), and non-muon primaries with their showers, per sector, plus step and bar
multiplicities. The primary muon carries most of it and most events have muon-origin deposits only; the non-muon part
is a minority that varies by sector. The numbers are reproducible with the subcommand and are not quoted here.

A straight-line muon model with in-bar energy-loss fluctuations covers the muon and muon-descendant parts directly. Delta rays mostly
deposit in the bar where they are made, so the per-crossing visible energy has to come from a Landau/Vavilov-type
distribution, not a mean dE/dx. The non-muon part is the hard part (section 6, tier 1b). It also matters for
physics: electromagnetic deposits are a main source of fake CRV coincidences.

Live time (measured 2026-10-16 on `g4ref_v1`, 1 M generated events): CRY `timeSimulated` gives **15.59 us of cosmic
exposure per generated event** (15.587 s per 1 M), i.e. 312 us per accepted event. One accepted event per 312 us of
live time means the 2026-10-15 profile's 0.45 s per accepted event costs about 1,400 CPU-s per second of exposure,
i.e. about 58 core-days of stock G4 per hour of cosmic exposure. (The earlier
0.62 s print was one subrun of the 40 k profile job: 15.5 us per event, consistent.)

### 3.4 Where the cut goes

Geant4 is used only to make `CrvStep`s. CRY, the CRV geometry service and the whole response chain are plain C++
(`CRVResponse` links no Geant4). So the fast simulation writes:

1. `mu2e::CrvStepCollection`: bar, Birks-corrected visible energy, start and end time, position and momentum, path
   length, a `Ptr<SimParticle>`, one entry per particle per bar crossing;
2. a consistent `SimParticleCollection` / `GenParticleCollection` with `CosmicCRY` primaries;
3. empty `StrawGasStep`, `CaloShowerStep`, `virtualdetector`, `SurfaceStep` and `MCTrajectory` collections (an
   `MCTrajectory` for each primary muon if the EventNtuple CRV-plane truth is wanted), plus `CosmicLivetime` and
   `GenEventCount`.

Everything from `CrvPhotons` on then runs unchanged. That is the A/B anchor for validation.

## 4. Performance goals (agreed 2026-10-15)

The baseline is one GPVM core running the stock chain: 0.43 s G4 + 0.02 s CRV response = **0.45 s per accepted
event**, or about 8,000 accepted events per core-hour. Goals are stated per accepted event, the unit that matters to
an analysis.

| ID | Goal | Target | Stretch | Why this number |
|---|---|---|---|---|
| P1 | Step generation (CRY + rejection + transport + CrvSteps), one CPU core | **<= 2 ms** per accepted event (>= 200x the G4 stage) | <= 0.5 ms | G4 on accepted events alone is 29 ms. P1 must beat that by >= 10x, so it does not win only through rejection |
| P2 | CRV photon + SiPM-charge stage on one A100 | **>= 50x** one CPU core running `CrvPhotons`+`CrvSiPMCharges` (>= 2,500 accepted events/s/GPU) | >= 200x | removes the response as the new bottleneck after P1 |
| P3 | End to end on one Polaris node (4x A100 + 32 cores), CRY to `CrvSiPMCharges` | **>= 1 M accepted events per node-hour** (~130x one GPVM core) | >= 5 M | a `kkts_extrap_v04`-sized sample (17.4 M events) in <= 18 node-hours, under one day on the reservation |
| P4 | Response-only re-run (one BO point) on 1 M cached accepted events | **<= 15 min** on one Polaris node | <= 3 min | makes the CRV-only alignment/response BO loop interactive |
| P5 | Container | one image, no CVMFS, cold start <= 5 min on a Polaris node, < 5 GB | < 2 GB | Polaris has no CVMFS ([apptainer page]) |
| P6 | Reproducibility | same seed + same backend gives bit-identical output; CPU and GPU agree statistically (section 5) | CPU/GPU bit-identical | needed for the A/B harness |

Where the stock digitisation (`CrvWaveforms`, `CrvDigi`) or reco must run, it runs on the CPU cores of the same
Polaris node inside the Offline image. Those two modules cost 0.15 ms per event, so they are not a bottleneck.

## 5. Validation (thresholds agreed 2026-10-15)

**Reference sample (decided before running).** A fresh G4 sample with the same Musing and the same CRY primaries
(same seeds): 1 M generated events, about 49 k accepted and about 260 k primary-muon bar crossings. It runs as 4
GPVM jobs on 4 hosts, about 3 h. Per-crossing statistics then resolve 0.1 % on a mean, so every threshold below is a
modelling tolerance, not a statistics limit. The 500 M-event production `dts.mu2e.CosmicCRYExtracted.MDC2025as`
(GenParticles and CrvSteps kept, on tape) is a second, larger reference, and the training set for tier 1b/2.

Comparisons go by truth-particle identity, never by time or distance pairing, and cover all 11 sectors. Rates are normalised per generated event (and per second of live time once 3.3 is resolved).

| Level | Observable | Threshold |
|---|---|---|
| V0 geometry | every one of the 576 bar placements and dimensions against `GeometryService` (v04) | <= 1 um, exact bar index |
| V0 generator | CRY primaries identical to the G4 job's `GenParticles` for the same seed | exact |
| V1 crossings (primary muons) | crossings per accepted event per sector | within 1 % |
| V1 | path length per crossing: mean, and distribution in the 5-95 % range | mean within 0.5 %; binned ratio chi2/ndf <= 1.5 (was +-2 %; changed 2026-10-16) |
| V1 | visible energy per crossing: MPV, mean, and distribution in the 5-95 % range | MPV and mean within 1 %; binned ratio chi2/ndf <= 1.5 (was +-3 %; changed 2026-10-16) |
| V1 | crossing time and position | time bias <= 0.1 ns; position bias <= 0.1 mm |
| V2 response, same digi on both arms | `crvpulses` PE per pulse, per sector: median, and spectrum over 5-150 PE | median within 2 %; binned ratio within +-5 % |
| V2 | per-bar pulse rate over all 576 bars (events with a pulse in the bar, counted once per event and bar; changed 2026-10-16 from per-SiPM pulses, which are ~4 correlated entries per crossing) | chi2/ndf <= 1.5 against G4, no bar beyond 5 sigma |
| V2 | coincidence clusters per generated event, per sector, at the data operating point (EX 3/8 PE/20 ns, T1/T2 4/8/10) | within 1 % |
| V2 | EX per-layer pulse efficiency for tracks into EX; module-gap dead width D per gap (`exdenom`) | layer efficiency within 0.3 points; \|Delta D\| <= 0.1 mm (the method's own systematic) |
| V3 non-muon (tier 1b) | non-muon share of visible energy per sector; non-muon coincidences per generated event | within 10 % relative; otherwise reported as a known deficit |
| V4 GPU response against stock `CrvPhotons`/`CrvSiPMCharges` on identical CrvSteps | photoelectrons per SiPM per event; arrival-time distribution | mean within 0.5 %; time quantiles (10/50/90 %) within 0.2 ns |
| Guard rail | fast-sim minus G4, for every V2 metric | <= 1/10 of the present MC-data difference in that metric |

V2 is run with two candidate arms through the same stock digitisation (`crvfastsim/README.md`, "V2 arms"): the fast
sim, and G4's own steps cut to the primary muon. The second isolates the tier-1a closure from the tier-1b deficit;
V2 proper (against full G4) needs tier 1b.

Tier 1b status (2026-10-19): the library (`crvfastsim/tier1b.py`, `crvfastsim/README.md`) is built from 20 production
dts files and tested on held-out reference shards; **V2 against full G4 passes in every sector group** (PE median,
PE spectrum, clusters per generated event, per-bar rates). Known limits: non-muon showers keep the bar indices of
their G4 event (they do not follow a geometry move); in the taggers, channels lit only by secondaries are about half
of G4's (within the thresholds). The EX layer-efficiency and gap-width rows of V2 are not computed yet.

Pass means every row passes. A failure is investigated, not tuned away: no free scale factors to force agreement,
because the response parameters are what the downstream BO fits.

## 6. Architecture

```
CRY (C++ lib, standalone)  ->  primaries (HDF5)
   -> fast transport (JAX; CPU or GPU)
        tier 1a: muons: ray vs material boxes (GDML export of extracted v04), Bethe-Bloch mean + Landau/Vavilov
                 fluctuation, Highland multiple scattering, Birks, ray-box bar crossings
        tier 1b: non-muon primaries + muon-induced showers: library resampled from G4 (built once from the production
                 dts), or a generative model trained on it
   -> CrvStep table (HDF5/Parquet, schema = CrvStep fields + particle table)
        |-> path A (validation, full chain): art source/producer `CrvStepsFromTable` -> stock Offline digi, reco, EventNtuple (CPU, Offline image)
        |-> path B (fast, tuning): JAX CRV response -> SiPM charges -> stock CrvWaveforms/CrvDigi, or a JAX port of them
   tier 2: replace the CRVConditions v6_0 lookup tables with a neural photon field (SIREN-style), trained on the
           tables, then on Celeritas/Opticks optical output; differentiable, so response knobs can be fitted by gradient
```

Design choices and reasons:

- **The interchange is a CrvStep table, not art.** JAX code cannot write art files. A small art module that reads the
  table and puts `CrvStep`/`SimParticle` products into the event keeps the whole stock chain as the reference (path A)
  and needs one Offline build.
- **JAX first** (PyTorch kept as a fallback). It runs on A100 natively, on CPU for GPVM and laptop tests, and possibly
  on Intel PVC. `vmap` over crossings matches the problem, and its autodiff serves tier 2.
- **Geometry from Offline, not re-typed.** Bar positions come from a dump of `GeometryService` (V0). The passive
  material comes from a GDML export of the extracted v04 world. Moving a module becomes a table edit (the CRV-only
  alignment needs exactly this).
- **Prior art reused.** Lookup-table and neural photon fields: SIREN for LArTPC photon libraries (arXiv:2211.01505),
  candela for IceCube (arXiv:2609.04695). Optical training data: Celeritas v0.7 optical offload (CUDA/HIP, needs the
  ORANGE geometry for surfaces) or Opticks (NVIDIA only). Muon flux: CRY as in production. EcoMug is a header-only
  cross-check.
- **Containers.** `crvfastsim` image (Python + JAX CUDA + CRY, < 2 GB). For path A, an Offline image built by a CVMFS
  trace-and-copy route (no Geant4 data needed, which the sim/digi trace still has to confirm).
  The R1 cvmfsexec route needs FUSE and user namespaces on Polaris compute nodes; ask ALCF before relying on it.
- **Submission.** IRI `launchJob` on Polaris with `pre_launch`/`post_launch` scripts (no `#PBS` lines); inputs and
  outputs on Eagle; Globus for the FNAL <-> ALCF transfers. Jobs are Python driver scripts in `iri/`, versioned.

## 7. Schedule (one month of hackathon access, from 2026-10-16)

| Week | Work | Exit check |
|---|---|---|
| 1 (10-16..10-23) | ALCF account and `GenesisHackathonOct26` membership; IRI token and a Polaris "hello" job; GPU and container smoke test; G4 reference sample (1 M) on 4 GPVMs; CrvStep table schema; geometry dump (V0); resolve live time | IRI job runs a JAX kernel on 4 A100s; V0 passes |
| 2 (10-23..10-30) | tier 1a muon transport + `CrvStepsFromTable` art module (user builds) | V1 passes; P1 measured |
| 3 (10-30..11-06) | JAX CRV response from the v6_0 tables; path A on the GPVM | V2 and V4 pass; P2 measured |
| 4 (11-06..11-15) | end to end on Polaris through IRI; scaling to several nodes; tier 1b first version; report and a decision on tier 2 | P3, P4, P5 measured; V3 status reported |

## 8. Risks and open questions

- ALCF compute-node policy on FUSE and user namespaces (cvmfsexec), and outbound https to the FNAL conditions DB.
  Plan for text-file conditions and the shrinkwrap image.
- Decided 2026-10-15: a tier-1b library built from one Geant4 run (or the MDC2025as production dts) is acceptable.
- The 1-month window covers stage 1 only. The tracker (stage 2) needs its own allocation (NERSC m4599 via HEPCloud,
  or LCRC).
- Lookup-table licensing/provenance: the CRVConditions v6_0 tables are Mu2e-internal. Keep them out of any public image
  or card.

## 9. Where the details live

- Baseline tools and results: `profile/` and the `crvfastsim` package (`crvfastsim/README.md`) in this repo.
- genesis-skills `amsc-skills/iri-api` (`references/ALCF.md`), `hpc-skills/pbs`.
- Offline: `CRVResponse/src/{CrvPhotonGenerator_module,MakeCrvPhotons}.cc`, `Mu2eG4/src/CRVSD.cc`,
  `CRVResponse/src/CrvStepsFromStepPointMCs_module.cc`, `GeometryService/src/CosmicRayShieldMaker.cc`,
  `Production/JobConfig/cosmic/ExtractedCRY.fcl`.

