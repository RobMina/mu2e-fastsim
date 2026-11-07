# Historical board observable and normalization

## Quantity to predict

The objective `flash.avg_trk_edep_per_gen_event_mev` is

\[
 Q = \frac{\sum_{e\in\mathrm{retained\ EarlyEleBeamFlash}}
                 \sum_{j\in\mathrm{StrawGasSteps}(e)} E^{\rm ion}_j}
                {\sum_{r\in\mathrm{generated\ SubRuns}} N_r}.
\]

The denominator counts **generated resampler events**, not individual electrons,
photons, retained detector events, or POT. One input event can contain several
particles and non-electron species. All tracker ionization in a retained event
contributes, including steps that did not themselves cause detector acceptance.
Zero-tracker events and SubRuns with zero retained events do not reduce exposure.

The central-value budget is **7.506758e-6 MeV/generated event**. The documented
POT conversion is 11.536718606512062 POT/generated event, equal to
25,000,000 / 2,166,994; divide Q by this factor to quote the documented per-POT
quantity. The budget originates from 6.50684e-7 MeV/POT times that factor, not
from either anchor's deposit. This is corroboration of the recorded conversion,
not a new inspection of upstream POT provenance. The BO constraint also uses
`k_sigma=1`: central threshold agreement is not optimizer uncertainty-based
feasibility. Missing uncertainty must not be replaced with zero.

## Historical selection, not current defaults

Saved shard-0 `config_dumper -P` outputs for both anchors establish:

- `genCounter` precedes resampling and selection; `g4consistentFilter` accepts
  status at most 9 before the detector filter. Failed/truncated simulation is
  not silently redefined as a physical zero or removed from exposure.
- `EarlyPrescaleFilter.nPrescale=1`; the early detector filter has **no time cut**.
  The main stream's strict fmod predicate does not define this objective.
- Detector acceptance is tracker OR calorimeter OR CRV. Tracker and CRV require
  at least one qualifying aggregated step for a particle, at most 1,000,000;
  ionizing tracker or visible CRV energy is strictly greater than 1e-6 MeV and
  momentum is strictly between 0 and 1e6 MeV/c. `KeepPDG=[]` accepts all species.
- Calorimeter steps require positive Birks energy and momentum strictly between
  0 and 1e6 MeV/c; qualifying energy is accumulated per SimParticle in float32,
  and a particle sum between 0.01 and 1e6 MeV inclusive accepts the event.
- `StrawGasStepMaker.KeepDeltasModule=""`. Offline v13_38_00 defaults
  `CombineDeltas=true`, so eligible short same-straw delta rays are combined
  with their parents. Do not substitute today's `g4run` exclusion or threshold
  individual raw StepPointMCs as if they were StrawGasSteps.
- Calorimeter production uses `deltaTime=0.2 ns`, `compressData=true`,
  `numZSlices=20`; CRV aggregation removes neutral particles. Compression keeps
  the detector-step collections without further step compression and retains
  all ancestry generations (`keepNGenerations=-1`).

Production's `SimJob MDC2025ax` backing is Offline v13_38_00. Its filter and
aggregation source confirms defaults omitted from the persisted configuration.
Only shard zero of each anchor was inspected; this is not a configuration census
of all 200 files or a reproduction of all random initial skips.

## Source and timing

Both saved configurations use the same
`sim.mu2e.EleBeamCat.Run1Baa.001430_00003251.art` input, `readMode="sequential"`,
`wrapFiles=true`, and `MaxEventsToSkip=2167248`. That configured skip maximum is
**not** the separately inspected pool count 2,166,994 and must not replace it
in the POT factor. The mixer chooses one initial random skip and then one
secondary input event per generated event. Contiguous resampling and wrapping
are not independent draws with replacement; uncertainty calculations must
respect source reuse and job/SubRun clustering.

`ProtonTimeOffset` uses `pulseType="default"`, `limitingHalfWidth=125 ns`, and
`potTimingDistribution_20160511.txt` plus
`acDipoleTransmissionFunction_20160511.txt`. One common offset is added per
event to generator/SimParticle global and StepPoint times, not proper times.
No early-stream time cut is present, but this does not authorize dropping pulse
handling from a future event-level chain that also includes timed aggregation.

## Independent saved-anchor cross-check

The original `anakit/flash/anakit_result.json` records identify analysis `edep`,
build `3ba8d23bbf47`, adapter `anakit-adapter/2`, and `SimJob MDC2025ay` (Offline
v13_39_00 backing). Both jobs succeeded over 100 input files. EdepAna sums all
retained StrawGasStep ionization with weight one and divides by GenEventCount.
The pinned analyzer adapter reads these averages from the endJob summary.
Its separate default `event_calo_edep_vis > 50` selection changes selected-event
counts only, **not the tracker-deposit average**.

| Anchor | Generated events | Retained events | Tracker ionization, MeV | Q, MeV/generated event | Receipt Q (printed precision) | Central budget |
|---|---:|---:|---:|---:|---:|---|
| bpzup21 | 11,000,000 | 14,407 | 40.74233601982624 | 3.703848729075113e-6 | 3.70385e-6 | Pass |
| bpzup12 | 11,000,000 | 53,729 | 149.9907398049316 | 1.3635521800448327e-5 | 1.36355e-5 | Fail |

Full-precision totals come from saved independent ancestry-analysis JSONs;
finished dumper counter logs report 11,000,000 events in 100 SubRuns each. The
anakit receipts and their successful logs agree at printed precision. No raw
files were recounted for this cross-check, no uncertainty inferred, and no
fast prediction or fitted scale used. The calo >50 selected counts (9 and 39)
are unrelated to the early-stream retained counts in this table.

## Reusable reduction contract

`flash_board_observable.summarize(retained_events, straw_steps, generated_subruns)`
implements the equation with exact integer identities and independent SubRun
counters. Inputs must already be aggregated, selected records: event keys;
step keys plus `ionizing_mev`; SubRun keys plus `generated_events`. It returns
per-event tracker totals, Q, retained-event diagnostics and the central budget
decision. Duplicate identities, orphan records, masked columns, invalid
energies/counters, retained counts exceeding generated counts and nonfinite
totals are rejected.
SubRuns with no retained events are preserved. No thresholds, time mask, POT
conversion, photon weights, or uncertainty correction are applied internally.

The separate [board selector](flash-board-selection.md) now supplies the historical
detector OR on complete already aggregated product tables and composes it with
this reducer. It preserves per-collection grouping and sequential float32 calo
sums. It does not supply raw-step aggregation, a fast response or upstream gates.

This is an observable reducer, **not** a detector selector, raw-step aggregator,
source sampler, response model or physics-validation result. Completeness of the
provided counter ledger remains a caller obligation; the function cannot infer
an entirely missing SubRun. It is not yet wired to the inclusive raw-step export
or the source predictor. Its synthetic anchor-arithmetic tests do not recount
or validate detector transport. Floating sums are ordinary, order-dependent
float64 reductions; central decisions use Q <= budget without a hidden rounding
tolerance. Rearranging deposits can change the last bit and flip a decision
within one ulp of the threshold; a regression demonstrates this limitation.
There is no claim of exact summation, order invariance, or bitwise EdepAna parity.

## Remaining end-to-end comparison

Apply the historical aggregation and event-level detector OR to a matching
source/transport chain, then reduce all retained tracker deposits over the
complete generated-event exposure. Preserve non-photon and non-target residuals,
unknown ownership, zero-response events, and shared ancestry correlations. A
weighted per-photon expectation alone cannot reproduce nonlinear event selection.
The direct TS3 inclusive reference is not yet this selected/aggregated observable.
Do not multiply it by a fitted anchor factor. Finite charged-foil transport,
remaining variable material, source spectra and response completeness remain
separate prerequisites for the two-anchor and 66/unseen-design predictions.
