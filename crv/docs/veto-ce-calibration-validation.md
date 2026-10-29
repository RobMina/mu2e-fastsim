# CE calibration validation and uncertainty admission

## Status and decision

**Design only; no new diagnostic, fit, extraction or response run.** This contract
follows the completed semantics, keyed joint tables and saved-table summary in
[veto-ce-proxy-semantics.md](veto-ce-proxy-semantics.md) and
[veto-ce-joint-diagnostic.md](veto-ce-joint-diagnostic.md). Reuse those products;
none needs regeneration. The existing 91,226 retained events and 42,319 selected
tracks are development evidence, not an independently validated CE target.

The common proton-count selection marginals are approximately flat, but the
observed lowest-minus-highest MC-PBI-quintile contrasts are 4.042 percentage
points at count zero and 4.499 points at count one. These descriptive contrasts
do not establish significance, causality or a calibration. One-variable
nProtonTCs weights and mean-intensity rescaling of F remain inadmissible.

Two distinct next steps are required: a bounded saved-row composition diagnostic
specified below, and external evidence identifying a calibration/validation
population. The former cannot substitute for the latter. C4 stays stock;
historical F and its conditional errors remain unchanged.

## Three different estimands

| Object | Numerator / denominator | What current products establish |
|---|---|---|
| Retained event-selection fraction | Events with at least one frozen window-selected track / all retained output events, A/N | 42,319/91,226; conditioned on upstream trigger/output selection |
| Generated-CE selection efficiency | Generated CE events satisfying the specified selection / all eligible generated CE events | Not identified: the required generated population and inclusion law are missing |
| Selected-track veto loss F | Vetoed selected-track weight / total selected-track weight in a specified target population | Historical result only, conditional on retained library/tracks and historical exposure |

For the future selected-track target, an explicit form is

```text
F_a = E_target[sum_{j=1}^{M_e} V_a(e,j)] / E_target[M_e]
Delta = F_fast - F_stock
```

Here M_e counts tracks passing the frozen pre-veto selection, including zero-track
events; V_a is the complete card-defined veto indicator for track j in arm a,
including event-gate failure. Multiplicity is retained, not replaced by an event
indicator or a best track. Equivalent weighted sums require a separately justified
target/inclusion law; this equation supplies no weights. An event-loss target would
be a different estimand and must not silently replace the track target. This is a
future target definition, not a recomputation of the historical grid estimator.

The target must bind the joint law of background mixture, intensity, bunch phase,
selected track time, signal contribution and conditions. The size-biased primary
law in [veto-exposure.md](veto-exposure.md) does not determine that joint law after
trigger and track selection. Independently pairing marginal track times and
background frames requires validation; an A/N regression does not provide it.
Any importance-weight route additionally needs source and target laws, overlap,
weight-uncertainty propagation and an audited denominator. None is admitted now.

## Calibration strategy before any fit

1. **Name the target and reference.** Specify intended run/conditions, CE population,
   pre-veto selection, timing convention, inclusion probabilities and whether the
   target is retained-output conditional or generated CE. Identify a calibration
   reference with independently justified intensity/timing information and known
   selection. No such reference dataset is selected by this document. MC truth
   from the current ledger is a diagnostic label, not an independent data reference.
2. **Define usable measured inputs.** nProtonTCs is a raw stopped-muon-activity proxy,
   not incident PBI. Inclusive tracker/calo counts and integer energy require
   signal-contribution, response and selection validation. Constant nprotons,
   pbterr and CAPHRI validity provide no fitted information here. Stored PBT in
   this named MC recipe is truth-smeared, so it is not admitted as an independently
   calibrated detector timing feature merely because it is under evtinfo.
3. **Bind deployment and transport.** Require historical producer/worker overrides,
   product tags, conditions/IOVs, trigger/readout windows, selection implementation
   and fit-time reference for the calibration and evaluation samples. One registered
   recipe/source chain and the stored ntuple version do not certify all workers.
   Establish which measured features are available under the intended deployment
   and how signal contamination and differing conditions are handled; do not subtract
   guessed CE hit counts or bunch time.
4. **Calibrate distributions, not only means.** A proposed measured-input conditional
   model must retain fluctuations and relevant joint dependence affecting nonlinear
   mixing/veto response. Predicting mean PBI alone is not sufficient. Name the model
   output and its bridge to the target joint law before choosing a model family,
   objective or weights. Truth-labelled models remain diagnostics; a deployable
   response/correction cannot take true PBI or true time as an input.
5. **Freeze development and validation.** Use the separation contract below. Choose
   features, binning/model family, tuning procedure, overlap requirements, relevant
   conditional checks and uncertainty method on development material only. Any
   future fit must have an explicit execution/resource contract of its own.
6. **Validate the loss functional.** Check predicted distributions, selected-time/
   occupancy dependence and the paired full-stage veto loss under the same target,
   conditions and card. Calibration agreement or small coarse-histogram TV alone
   cannot certify F. Freeze an absolute F/delta tolerance in percentage points,
   uncertainty-aware equivalence rule, coverage requirements and multiplicity of
   checks before viewing new validation outcomes. These numerical tolerances are
   not chosen here and must not be copied from the observed historical delta/error.

Failure to identify the target/reference, deployment, independence or validation
population blocks a calibration/promotion claim, not just its error bars.

## Validation separation and leakage

All eleven retained sources have already contributed to published summaries.
Randomly splitting their rows now, leaving one file out, or reserving a source
index after inspecting those results does **not** create an untouched holdout.
Such splits may be development stress tests only; none is created in this step.

Before allocating future train/tune/validation samples, construct an auditable
physical-dependency map: original generation, resampling/recovery ancestry, shared
library histories across species, mixtures and downstream replicas. Connected
histories stay in one partition. File names, seeds and disjoint immediate parents
alone cannot establish separation. Shared histories can connect multiple mixtures;
the dependency structure is not necessarily a simple nested tree. If it collapses
available material into one connected population, do not invent independent folds.
Conditions/run-domain holdouts are additional transport tests, not replacements
for ancestry separation. A common fixed library permits at most a clearly labelled
conditional validation, not independent-library validation.

A genuinely unexamined validation population must be separately identified and
sealed before model choices. Its manifest must record target, identities, physical
groups, assignment rule, hashes, prior access and permitted evaluation. No present
split or independent-validation sample is claimed. Opening the validation result
is a one-time decision stage; a failed gate is not rescued by retuning on it and
calling the same material held out again. Calibration-reference and final-validation
roles must also be separated, even if both have independent reference measurements.

## Uncertainty units and what may be reported

| Level | Required unit / dependency | Current status |
|---|---|---|
| CE observations | Full event identity; tracks within an event stay together | Identity is known; physical independence and target inclusion law are not |
| File/clock composition | Actual source index and stored-PBT stratum | Descriptive sensitivity only; eleven files are not eleven independent exposures |
| Mixing variation | Original mixed-frame draw under a frozen library/exposure law | Conditional on that library; mixtures may share upstream histories |
| Response variation | Keyed response replicas nested within each original frame | Replicas are not additional exposure or independent library samples |
| Fast-stock difference | Paired arms on common raw mixtures, compatible target/card | Preserve pairing and covariance; do not add arm errors as independent |
| Finite-library variation | Original physical histories with ancestry-aware joint reuse across species | Unknown; library uncertainty remains null, not zero |
| Calibration and target law | Reference, fitted parameters, IOV/transport and target-estimation dependencies | Unestimated; propagate jointly with affected selection/response when admitted |
| Model/timing discrepancy | Explicit alternative contracts and closure evidence | Not a random error bar unless a justified probability model is supplied |

A finite-library design must resample upstream histories with the intended reuse
law **before nonlinear mixing**, then pair arms and nest response replicas. It
must preserve shared-history dependencies between mixtures/species. Bootstrapping
final frames, filenames, aliases, tracks or response seeds cannot stand in for
this design. Repeated stock values paired with several fast seeds are not several
independent stock observations. The observed group-weight concentration is not
an effective independent-history count. See
[veto-physical-reuse-gates.md](veto-physical-reuse-gates.md).

Conditional event/frame sampling errors, when their assumptions are justified,
must be labelled with what is held fixed. Calibration, mixing, response, library
and target-law uncertainties cannot simply be added in quadrature without their
joint dependency model. Sparse-support counts do not certify precision. This
step adds no binomial errors, bootstrap intervals, significance tests, total error
or uncertainty estimate; missing components remain explicitly unestimated.

## Next bounded diagnostic: source/stored-PBT composition

**Specified, not implemented or executed.** Its question is whether the retained
endpoint association changes when low/high MC-PBI events use the same *observed*
source/stored-PBT composition. It is not a calibration or causal adjustment.

### Saved inputs and accounting

Use only the saved numeric proxy rows, frozen selected columns
`run/subrun/event/t_mid/p_mid`, and old seven-column exposure ledger. Require the
original selection and rows reports; never open ROOT or selected `coincs`.

| Binding | SHA-256 |
|---|---|
| Original selection report | `f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3` |
| Successful rows report | `af675694a4a997295d42d84a98bc8fb0aedac160aa83c2cc69002e3cb4e2b2e8` |
| Numeric rows | `20f012be65a9487226923f0388a72f06eca8ee02c95bcdcadc822b8b68f45f3a` |
| Selected Parquet | `7f60b88899f35bea7cf0a438aa18250ff0021ad229bc3c000bfd180fa386a2ba` |
| Exposure ledger | `b9f94c948099cdeddb1e27634bb4f2f232619aa11d854ae5a96d4d5339e955cf` |
| Completed joint report | `09c8831caca851de8ebb78cee91f467d9ff9f7c6b346ebdffe7ca5a2c70e3831` |
| Completed saved summary | `63fc77886c3febb00ba6120ee9dfd6140bb3fedf12260312544f3bb574b68e0a` |

The original joint tables lack the complete source-by-proxy-by-PBI-by-PBT
cross-classification; marginal summaries cannot reconstruct it. Reuse keyed rows
for that new cross-classification, not a ROOT re-extraction or an old consumer
rerun. Preserve the original full signed-int64 tuples, source basename indices
(including missing index 7), strict 85 < p_mid < 200 cut, list-position tracks,
all zero-track events and global/matched anti-join accounting. Reconcile inherited
counts and marginals to the pinned completed reports before publishing anything.

### Frozen descriptive comparisons

- Use exact raw proton counts c=0 and c=1, chosen because their endpoint contrasts
  were already observed. This is explicitly exploratory, not a preregistered
  independent hypothesis test. Keep other counts and middle PBI strata in the
  complete accounting; do not replace the endpoint denominator with all events.
- Use the original MC-PBI edges `[2412587,11844973,15682329,19931318,26354159,93225507]`:
  lowest quintile is left-closed/right-open; highest includes its final edge.
  MC PBI is truth diagnostic only; do not refit quantiles within files/counts.
- Blocks b are actual source index crossed with stored-PBT edges
  `[-240,-230,-220,-210,-200,-190]` ns, left-closed/right-open with both flows.
  Preserve empty cells and the full source axis. No adaptive merging or rephasing.
- For each c,b,endpoint k record N/A/T, track multiplicities and raw selected-time
  counts in the original 0..2000 ns, 100-ns bins with flows. Histogram totals equal T.
- Define common-support blocks B_c by **N_c,b,low >= 100 AND N_c,b,high >= 100**.
  This rule uses only event denominators, not A/T. It is a descriptive bookkeeping
  threshold, not a precision or independence certificate. Existing shape flags
  A>=20 and T>=30 are reported separately and never determine B_c or its weights.

For each count separately report these three contrasts, in percentage points:

```text
D_all,c = 100 * (sum_all A_low / sum_all N_low
                - sum_all A_high / sum_all N_high)
D_common,c = 100 * (sum_Bc A_low / sum_Bc N_low
                   - sum_Bc A_high / sum_Bc N_high)
q_c,b = (N_c,b,low + N_c,b,high) / sum_Bc(N_c,b,low + N_c,b,high)
D_standardized,c = 100 * sum_Bc q_c,b * (A_c,b,low/N_c,b,low
                                      - A_c,b,high/N_c,b,high)
```

D_common separates support restriction from the subsequent composition change;
comparing D_all directly to D_standardized would conflate them. Each count has
its own B_c and q; do not pool them or treat their standardized values as the same
target population. q describes a retained endpoint mixture, **not deployable
correction weights**. No common support gives null standardized contrast, never
zero. Every zero-denominator fraction is null.

For each count/endpoint, retain total, supported and excluded N/A/T and loss
fractions relative to that endpoint's own totals. Publish every cell, unsupported
reason, per-source contrasts and common weights, not just an average that could
hide cancellation. Selected-time shapes remain selection-conditional; no event
selection probability versus track time exists for unselected events.

The label is **source/stored-PBT composition-standardized retained association**.
Source is not ancestry; stored PBT is an MC surrogate, and coarse bins/flows do
not fix within-bin differences. Neither attenuation nor persistence of the
contrast proves causality, sufficiency, factorization, lack of signal contamination
or transportability. The full retained denominator remains trigger-conditioned.

### Implementation admission and stop rules

A separately reviewed implementation must preserve the saved-input caps and
vectorized joins from the completed joint contract: selection <=64 MiB, ledger
<=8 MiB, numeric NPZ <=16 MiB, reports <=8 MiB, footer <=1 MiB, declared selected
column bytes <=256 MiB, <=2.1M selected rows/tracks, <=64 tracks/row and <=100,000
ledger events. Proposed new JSON cap is 2 MiB; failure is not permission to drop
cells. These are not whole-process RSS or malicious-decoder allocation guarantees.
Require input/source pins, before/after checks, exclusive report-last publication,
synthetic identity/edge/flow/support/conservation tests and fresh host/resource
admission before execution. This document does not authorize that execution.

Pin, identity, join or inherited-count mismatches fail closed: no alternate input,
deduplication, replacement extraction, bin tuning or support relaxation. Insufficient
support yields only the supported diagnostic and explicit exclusions, not a broader
claim. No inferential interval or calibration fit is part of this diagnostic.
`calibration`, `weight_model`, `F`, `finite_library_uncertainty` and total uncertainty
remain null. A further independent validation/reference decision, physical reuse
contract and full-timing admission are still necessary for veto closure.
