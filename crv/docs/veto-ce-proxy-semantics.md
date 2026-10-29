# Retained CE luminosity-proxy semantics and diagnostic scope

## Evidence and limits

This follow-up reuses the completed eleven-file availability audit. It traces the
registered digitization recipe upstream of the CE ntuple, rather than assuming the
ntuple release also made the detector observables. No production configuration is
executed, no reconstruction is repeated, and no calibration function is fitted.

For retained ntuple sequencer `001430_00000000`, catalog metadata gives:

- Ntuple parent: `mcs.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.001430_00000000.art`.
- That file's parent: `dig.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.001430_00000000.art`.
- Both declare 8,212 events, consistent with the retained ntuple. Their catalog
  checksums/counts are not fresh physics-file hashes or generated-CE denominators.
- The digitization job definition is
  `cnf.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.0.tar`, SHA-256
  `b747fde8badafff79918a20e6a6bf2ea534351c549ec6a73e6d7b1dffd3dc1bb`.
  Its complete 254,851-byte job parameters and 519-byte FCL name
  `SimJob/MDC2025au/setup.sh`, backed by **Offline v13_33_00**.
- The FCL includes `mixing/Mix.fcl` and `mixing/OneBB.fcl`, uses `Sim_best/v1_3`
  conditions and SDF 0.8. The previously captured ntuple recipe instead names
  `AnalysisMDC2025/v02_01_00`, whose inspected backing chain reaches v13_35_00.

These are registered-recipe and selected release-source observations. One catalog
ancestry chain and stored version 6.13.2 do not certify all workers, overrides,
product provenance, calibration IOVs or the complete transitive FHiCL expansion.
The source capture is bounded and hash-checked before/copy/after. Current external
analysis source snapshots are labelled available source, not historical code pins.

## What is actually stored

EventNtuple's captured `prolog.fcl` names `TrigFlagPH`, `TrigMakeSH` and
`CaloHitMakerFast`; `InfoStructHelper::fillLumiStreamInfo` copies their intensity
products without a PBI calibration. Missing products reset the corresponding
counts to -1. The completed audit found no such resets in the retained files.

### Proton-candidate count: not beam protons

In the digitization recipe's `core/producers/trigTrkProducers.fcl`, `TrigFlagPH`
is **TTDeltaFinder**, using `DeltaFinder`, not `FlagBkgHits`. The latter's counting
algorithm must not be substituted merely because it also writes the product type.

`CalPatRec/src/DeltaFinder_module.cc` increments `np15` for each reconstructed
proton candidate with `pc->nHitsTot() >= 15`, then writes
`IntensityInfoTimeCluster(np15)`. `ProtonCandidate` counts combo hits separately
from contributing straw hits. The increment occurs before the subsequent
single-station 4-keV check used in hit flagging: that later check does not reduce
`np15`. The finder input is `TrigMakePH`, with single-straw input `TrigMakeSH`.
Nominal finder parameters include 40-ns bins, maximumTime 1750 ns, and proton seed
and hit energy thresholds of 0.003 MeV. Those describe the inspected configuration,
not a rectangular acceptance window or a full algorithm specification.

This count is a **raw reconstructed proton-candidate activity proxy**, documented
in source as a proxy to stopped-muon rate. It is not the incident beam-proton count.
No conversion to beam PBI is supplied by these copying/counting modules. Selection
and detector occupancy can affect it; the CE contribution/misclassification has
not been measured. `nProtonTCs` is the primary diagnostic candidate, not an approved
calibrated correction variable.

### Tracker hit count includes signal activity

The final `trigDigiInputsEpilog.fcl` selects `TrigMakeSHFromDigi`, a `StrawHitReco`
instance with `FilterHits:false`. `StrawHitReco::produce` sets the intensity count
to the size of its produced single-straw ComboHit collection. Nominal time,
energy/radius selection flags do **not** become filtering cuts merely because
those thresholds exist: `StrawHitRecoUtils` retains hits when filtering is false.
Its times subtract the ProtonBunchTime offset before subsequent time flags.

Thus `nTrackerHits` is inclusive detector activity from the input digis, not an
independent beam monitor or a count of accepted-track hits. CE-associated digis
can contribute. Neither removing an estimated CE hit count nor fitting a
single-variable correction is justified here.

### Calorimeter hits and integer energy

The same final epilog selects `CaloHitMakerFastFromDigi`, using `CaloDigiMaker`,
5-ns sampling, pulse-combination tolerance 11 ns and nominal ADCToMeV 0.0625.
The module obtains pulse energy from peak minus estimated baseline and combines
nearby pulses per crystal. It processes the supplied digi collection without a
separate explicit event-time cut in `extractHits`; upstream digitization/readout
windows still matter and are not inferred from that absence.

- `nCaloHitsD0/D1` count reconstructed non-CAPHRI crystal pulses by disk.
- `caloEnergy` accumulates energies from **all** reconstructed crystal pulses,
  including CAPHRI. It is an `unsigned short` accumulator updated from float
  energies, then copied into EventNtuple's integer field. This entails integer
  conversion during accumulation; it is not a saved floating total or beam PBI.
  The declared energy unit is MeV, but the audit does not validate energy scale,
  rounding loss, overflow behavior or a luminosity calibration.
- CAPHRI energy cuts (nominal strict 1–5 MeV) govern the separate compact CAPHRI
  hit list, not the disk counts or total-energy accumulator.
- `caphriIsValid` is set true when the calorimeter intensity product is filled;
  it does **not** mean the event has a CAPHRI hit or passes a beam-intensity test.

These observables can include CE energy/hits. Their selection dependence is useful
to describe, but is not by itself a measurement of changing incident intensity.

## Bunch time and selected-track time

The declared chain uses `EWMProducer` for both EventInfo's ProtonBunchTime and
MC ProtonBunchTimeMC. In the inspected v13_33_00 CommonMC prolog:

```text
RecoFromMCTruth: true
RecoFromMCTruthErr: 0.5
```

`EventWindowMarkerProducer` computes the truth bunch time relative to the DAQ
event-window marker from conditions, initial clock phase and event number. With
this configuration it fills the reconstructed product by adding Gaussian noise
of width 0.5 ns to truth and sets the stored error to 0.5 ns. That matches the
constant error observed by the audit, but remains recipe/source evidence rather
than full runtime provenance. **The `evtinfo` namespace does not make this an
independently measured detector clock estimate in this MC sample.**

The selected-track adapter saves the first matching tracker-middle intersection's
`time` and momentum magnitude, in stored segment order; it does not average,
choose the earliest intersection, add/subtract bunch time, or wrap modulo a
period. EventNtuple's helper copies `KalSeed` intersection time directly. Tracker
and calorimeter hit reconstruction already subtract the ProtonBunchTime offset;
therefore a second subtraction must not be introduced by analogy. Until the
complete historical selection/fit path is bound, the diagnostic labels the
quantity **raw stored t_mid** and performs no additional time transformation.

A saved selection script explicitly requests `--cutset dev --no-gate`, with
500 input ntuples. Its saved successful log reports 1,930,163 event rows and tracks.
This improves invocation provenance but does not cryptographically bind the
historical imported `pyutils`/analysis checkout or each cut implementation. The
artifact's original SHA-256 and the saved strict **85 < p_mid < 200** report remain
the execution contract; no selection is rerun. The local adapter stores jagged
track lists per event despite its older opening docstring saying per-track rows.

## Decision for the next diagnostic

Proceed only with a **descriptive retained-sample joint diagnostic**:

1. Persist slim keyed rows for the already audited scalar arrays, comparing every
   consumed array digest/dtype/shape with the successful availability report.
2. Join full run/subrun/event identities to the frozen selection and old exposure
   ledger, retaining zero-selected-track events and all track multiplicities.
3. Compare raw proxy, raw stored bunch-time and raw selected-track-time bins.
   Truth PBI comes only from the old ledger and is explicitly labelled diagnostic.
4. Record per-source-file support, selection rates, multiplicities and time-shape
   counts. No fitted calibration, weights, correction, new F or significance claim.

The retained digitization recipe selects triggered or triggerable outputs; the
ntuple's own hasTracks/hasCRV flags are false. The denominator is therefore retained
output events, not all generated CE events. The analysis cannot demonstrate proxy
sufficiency or remove trigger/selection conditioning by renormalizing counts.
An independent calibration/validation contract and historical execution evidence
remain necessary before using a measured-variable response/correction model.
