# Historical EarlyFlash detector selection from aggregated products

`flash_board_selection.select` implements the historical detector OR on complete
**already aggregated** StrawGasStep, CaloShowerStep and CrvStep tables. It does not
aggregate raw StepPointMCs, choose the source, add the proton pulse, apply the
upstream G4 consistency filter, or predict detector response. It is not connected
to the inclusive exporter or the fast source navigator. Synthetic tests establish
this component's contract, not a two-anchor or 66-design physics result.

The target is Offline v13_38_00 `Filters/src/DetectorStepFilter_module.cc` with the
saved EarlyEleBeamFlash settings described in
[flash-board-observable.md](flash-board-observable.md). There is no PDG restriction,
no time cut, and no total-calorimeter-energy override in this configuration.
It is not the main-flash filter, and not anakit's separate calo >50 diagnostic.

## Inputs and completeness

All inputs are dictionaries of equal-length one-dimensional NumPy-compatible
columns. Identities must be exact nonnegative integers, including values above
2**53 and mixed signed/unsigned columns. Masked columns, ambiguous duplicate
identities, missing products, wrong detector roles and orphan records are errors.

- `events`: unique `run,subrun,event` identities for **all events reaching the
  detector filter**, including those rejected by it. This is not necessarily the
  generated-event universe: upstream gates may reject events before this filter.
- `products`: `run,subrun,event,product,detector,size`. Detector codes are
  `TRACKER=1`, `CALO=2`, `CRV=3`. Every event must declare each role, with explicit
  `size=0` for empty input collections. Multiple collections per detector are
  supported. Every configured input collection must be supplied; the function
  cannot infer a completely omitted additional collection or event.
- Each step table: `run,subrun,event,product,index,particle_product,particle_key`,
  `momentum_mev_c`, and `ionizing_mev` (tracker), `birks_mev` (calo), or
  `visible_mev` (CRV). Particle product/key identify the dereferenced SimParticle,
  not just a particle key shared across products. The caller must resolve valid
  pointers; no ambiguous/null-pointer state is represented by this schema.
- `index` is the original position in its detector collection, not an arbitrary
  export ordinal. Unique step identities, row counts equal to declared sizes,
  and indices in `[0,size)` enforce complete contiguous collection indices.

Energy and momentum columns describe the **stored accessors**, losslessly
representable in float32 (float64 exact promotions are accepted). The historical
step energy accessors and calo momentum are floats; tracker/CRV momentum is the
float-vector magnitude accessor, not a fresh double-precision norm reconstructed
from exported components. Input rounding is never silently substituted for the
historical storage/aggregation operations. Nonfinite, negative or inexact values
are rejected; integer energy/momentum inputs above 2**53 are rejected before a
floating comparison can hide integer rounding. The extractor remains responsible for obtaining those accessors
from the appropriate historical aggregation. In particular, raw G4 energy or
momentum columns do not satisfy this contract just by casting them to float32.

## Selection details that affect decisions

Tracker and CRV qualify steps using energy **strictly greater than** the double
constant `1e-6` and momentum strictly between 0 and `1e6`. Steps are counted per
SimParticle **within each input collection**. A count in `[1,1_000_000]` accepts;
counts are not added across collections or across distinct particles. Counts
above the historical 32-bit unsigned range are rejected, never wrapped into an
accepted multiplicity. All species are eligible; CRV neutral removal is an
upstream aggregation operation.

Calorimeter steps qualify with positive Birks energy and momentum strictly
between 0 and `1e6`. Qualifying Birks energies are accumulated per particle,
within each input collection, using **sequential float32 addition in original
collection order**. A particle sum in `[0.01,1e6]` accepts. This is not the total
event energy, a float64 sum, a tree/pairwise sum or a sum across collections.
The implementation groups exact identities, orders each group by collection
index, then uses unbuffered `np.add.at` on a float32 accumulator. There is no
per-event or per-particle Python loop. Nonfinite sums fail explicitly rather
than silently becoming a physical rejection.

Comparisons promote float32 accessor values/sums to float64, matching their
comparison with the C++ double thresholds. This matters: `float32(0.01)` lies
**below** the double constant `0.01`, and `float32(1e-6)` likewise lies below its
double threshold. NumPy's implicit comparison against a Python scalar can
instead downcast the threshold, changing the answer.

The overall decision is tracker OR calo OR CRV. The result contains boolean
`retained,tracker,calo,crv` masks in ledger order, a `retained_events` table, and
**all** tracker steps in those retained events. A subthreshold tracker step
still contributes to Q if any detector accepts its event. Retained events with
no tracker steps remain in the output ledger. Photon weighting is not part of
this API and is never applied.

## Observable composition

```python
from beamfastsim import flash_board_selection as selection
result = selection.summarize(events, products, straw_steps, calo_steps,
                             crv_steps, generated_subruns)
```

The independent generated SubRun counters include all generated exposure, not
just events reaching or passing the detector filter. Composition validates all
filter-input event memberships/counts against these counters, then reduces the
retained records with `flash_board_observable.summarize`. It adds the detector
masks and number of events reaching the detector filter. Zero-retained SubRuns
remain in the denominator. No uncertainty or optimizer feasibility is inferred.

The composition does not certify upstream completion. Failed/truncated G4 events
must remain visible to campaign validity checks; supplying counters does not
turn their missing physics into valid zero response. Ordinary float64 Q
reduction has the order dependence documented in the observable contract.

## Validation and remaining work

Tests cover individual detector OR branches, all retained subthreshold tracker
energy, zero tracker/deposit events and zero-retained exposure, strict energy
and momentum boundaries, the float32 lower-threshold distinction, per-particle
and per-collection grouping, sequential float32 rounding and row-order replay,
the million-step count boundary, exact large identities, product completeness,
masked/malformed inputs, overflow and composition with the complete counters.

Remaining work includes the actual historical raw-to-aggregated conversion
(short-delta combining in tracker, timed calo compression and CRV neutral
removal), an export of complete products and accessor values, a matched
source/resampler/pulse/upstream transport chain, and all-species fast response.
This module supplies the detector-selection stage only. No new G4 jobs, frozen
reference changes, fitted scale, production default changes or builds are part
of it.

The new [aggregated accessor exporter](flash-board-export.md) is prepared as a
separate read-only art satellite, with a typed Python reader and synthetic
fixture checker. It still needs the user's build and art-level validation.

A bounded inspection of existing extraction code found a concrete integration
gap: `FlashReferenceExporter` reads raw StepPointMC collections, not the three
aggregated detector products. The existing `StrawGasStepTableDumper` covers only
tracker energy/ancestry and lacks the complete product-qualified identities,
original collection indices and exact magnitude accessors needed here; stock
human-readable printers also omit fields or round values. None is a verified
adapter for this API. Do not synthesize product completeness from those tables.

The next data check can reuse existing art without rerunning G4: first validate
a complete aggregated-accessor exporter against historical selected anchor art;
then apply explicitly historical aggregation to a small frozen inclusive TS3
sample and retain its unfiltered products. The inclusive FHiCL retains raw art
and ancestry and the runner saves geometry, but this was a configuration/source
inspection, not a fresh art-product census. Matching geometry/field/conditions
and the missing source pulse remain requirements. Refiltering already selected
anchor art cannot test rejected events, and postprocessing the direct-source
sample is an aggregation diagnostic, not a board-normalized fast prediction.
