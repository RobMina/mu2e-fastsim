# Digi hand-off for stock trigger replay

**Current execution state:** R7 is paused by user as of 2026-11-05 06:10 EST; no
R7 jobs are running. The [pause/resume handoff](PAUSE_RESUME.md) records completed
S/F and first-parent C evidence, remaining assignments and gates. This contract
is not permission to launch the nine remaining C parents or full-history C trigger.

This contract is for a **combined, event-keyed art input**, not an ASCII step table, a reco ntuple or a
standalone tracker/calorimeter file. The first control replays the stock menu from embedded provenance;
a fast arm changes digi products, not trigger thresholds, conditions, output selection or the input population.

## Required products

| Product | Stock input tag | Required payload and invariants |
|---|---|---|
| `mu2e::StrawDigiCollection` | `makeSD` | StrawId, both-end TDC/TOT, firmware PMP and flags; repeated digis preserved; integer digital units, original order |
| `mu2e::CaloDigiCollection` | `CaloDigiMaker` | SiPMID, integer time origin in the release's persisted units (clock ticks in the checked release), integer ADC waveform, valid peak index; repeated pulses and both SiPMs preserved |
| `mu2e::EventWindowMarker` | `EWMProducer` | Original spill type and event-window length; never infer from a sample name or substitute a default CFO marker |
| `mu2e::ProtonBunchTime` | `EWMProducer` | Original bunch-time offset; used in both tracker and calorimeter trigger hit reconstruction |
| `art::TriggerResults` | `TriggerResults::Digitize` | Same-event reference decisions and embedded scheduled names/bits; must be retained through injection |
| Event identity | art event/run/subrun | Exact original identities and order; no renumbering after filtering or batching |

Geometry, tracker alignment/drift/electronics response and calorimeter calibration must match the selected
production release and conditions purpose/version. An empty **present** collection is an event with no digis;
a missing collection is a contract error, never an instruction to fall back to another process's product.
The original full collection is required for occupancy; hit truth-matching alone cannot certify this contract.

The production trigger's `FitType:5` uses firmware PMP; ADC waveforms are not required by that hit maker, but are
required if another configured fit type reads them. Calorimeter-trigger hit making is `CaloHitMakerFast`, not the
offline TemplateFit reconstruction. It uses waveform peak minus pre-peak baseline, then the sampling period and
bunch-time offset and channel offset. In the checked release the fast hit time is
`(t0 + peakpos) * digitizationPeriod - bunchOffset - channelOffset`, with `t0` in ticks; older files can carry
schema evolution from ns. Preserve the actual persisted units/version, not a filename-era assumption.
The separately audited older cosmic production release uses integer `t0_` in ns and computes
`t0 + peakpos * digiSampling - bunchOffset`; its digitizer writes a sampled time rather than a tick index.
Do not apply the newer tick conversion or channel-offset formula to that older menu. Persisted schema and
released implementation, not a shared path name, determine which contract applies.
A hit/cluster-energy proxy cannot stand in for these digi payloads.

## Arms, in order

1. **Stock control:** original straw + calo digis, timing, event IDs; complete preselection-parent history through one replay process, not just selected production digis.
2. **Tracker-only substitution:** new straw digi collection under an unambiguous tag, stock calo and timing.
3. **Calorimeter-only substitution:** stock straw digis, new calo digi collection, stock timing.
4. **Both:** both new digi collections; all other inputs and event selection unchanged.

Every arm must preserve events with zero digi output. Do not concatenate two art files and assume the framework
joins them on event ID. Produce/inject both required collections into one combined event, or use a separately
validated event-association mechanism. Separate input files with identical IDs are not a friend-tree join.

## Consumer tags to update

Use `manifest.json.external_bindings` for the known hit-maker/timing consumers. In addition, inspect the complete
embedded path modules: enabled `DigiFilter` instances use `strawDigiCollection` and/or `caloDigiCollection` even
when no hit reconstruction runs. `CaloCosmicCalib` instead consumes clusters and their hit pointers. Replacing only
`TrigMakeSH.StrawDigiCollectionTag` or only `CaloHitMakerFast.caloDigiCollection` leaves mixed stock/fast paths.
Use an explicit `label:instance:process` tag where both stock and injected products coexist. Reject explicit
old-process tags that accidentally read original reconstruction rather than the replay's products.

`trigger/replay.py` records known digi/timing bindings and rejects references to removed truth-decoration
producers or explicitly pinned old reconstruction. That is a scoped preflight, **not** a complete C++ consumes
graph. Product presence and actual consumed provenance still need a runtime check on each injected arm.

## Comparison and acceptance

For each original input shard, compare every event/path bit. Summaries include stock/fast pass totals, each
direction of migration and the paired efficiency difference. Errors are computed across paired shards, not
from independent binomial errors; held-out shards are used once after development choices are fixed.
Original output is trigger OR truth-selected, so these efficiencies are conditional on stored inputs, not
unbiased production rates or absolute trigger efficiencies. Full input order/state is needed for luminosity
buffering and the phi-prescaled `HelixFilter`, which advances a filter-call counter. Both are included in the
complete-history stock control: all 52 paths agree in accepted/state/`wasrun`/mapped endpoints over 28,044 events.
The earlier selected-stream differences remain documented, not ignored. Raw indices differ after path shortening. Aggregate PrescaleFilterFraction totals and buffered intensity
measurements cannot recover event-keyed phi-filter call history. A stock ordinal sidecar is not automatically
valid for a fast arm: altered upstream filtering changes which events reach the phi filter. Exact history needs
complete preselection coverage for each arm, or an independently validated state-restoration mechanism; never
fit counter offsets to the observed stock outcomes.

For the next matched hybrid input, return a receipt identifying the complete source shard set/order, geometry
and conditions, original retained decision process, zero-output event coverage, stock timing provenance and
explicit substituted digi tags. A historical response-only control or shower/step library is not that combined
art input. Validate the stock arm first under the [frozen procedure](VALIDATION_PROTOCOL.md); each fast arm must
advance its own counters over that same complete population.

Keep the release, full expanded trigger config hash, input-file checksum/event list, injected product tags,
producer config/seed, event counts and completed-job exit status with the private result. Timing is measured
in a separate quiet run with the same path schedule, no per-event printing and no warmed-input assumption.

Preparation can retag every known enabled direct-digi consumer without changing timing or trigger cuts:

```bash
python3 trigger/replay.py "$EMBEDDED_CONFIG" --straw-digi-tag FastStraw::Injected \
  --calo-digi-tag FastCalo::Injected --output "$PRIVATE_OUT/both"
```

Use either option alone for a single-detector arm. Tags require explicit producer, instance (possibly empty)
and input process; a replay-produced tag is refused. The private manifest records every old/new binding.
This does not inject products, synthesize missing digis or assert physics validity; the combined input must
already meet the product/event contract above, and unknown module types still need a consumes audit.
