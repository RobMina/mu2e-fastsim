# Saved pulse identity inventory

`veto.saved_pulse_inventory` inventories two existing response controls without
running a finder, fitting, optical/SiPM response, stock reconstruction or coverage.
It is a bounded admission inspection, not a coincidence result or production
certificate. The completed streaming-pilot and legacy nonempty audits are not
repeated.

## Evidence and bounds

Require external pins for both generation summaries and their completed saved
verification reports. For the all-live four-frame control, check all existing
atomic receipts, inventories and code snapshots, but decode only registry,
selection, calibration, full-frame pulse/waveform tables and small frame ledgers.
Partitions are already validated evidence, not additional input draws. For the
older extreme-time control, use its pinned output hashes and qualified event
census from the original optical summary. Check all listed outputs without
regenerating any prior report. All consumed metadata, pulse/condition tables and
executed code are rehashed at the end. No raw mixture, stock ROOT, LUT or selected
track payload is opened.

Parquet file cap 32 MiB, at most 64 columns/4,096 row groups and 128 MiB declared
uncompressed bytes before decoding; task row caps are 9,000 registry, 15,360
calibration, four frames, 2,048 pulses/waveforms per table and 20,480 conditions.
Caps do not guarantee RSS. Output is one exclusive atomic inventory, at most
2 MiB including receipts and source snapshots. No payload is copied into a new
physics sample.

## What is checked

- Immutable full-registry `(input_sha256,run,subrun,event,frame_id)` identities;
  older event-only rows gain source qualification only via their pinned full
  event census, never nearest-time matching or invented labels.
- Actual saved `wf,chan,starttdc,seq`, boolean fit/status flags and float64 peak,
  relative and reco times. Require the exact forward bunch-time relation.
- Native `(frame_id,wf,seq)` order equals structural channel/start/sequence order
  and stored row order; waveforms provide a bijective structural mapping. No
  sorting repair or invented waveform links is allowed.
- Ignored-channel flags equal authoritative calibration status bit 1. These
  diagnostic pulses are retained; any future finder-only validation must use
  the already-pinned policy that excludes ignored rows only from finder input.
- All-live and optical-union thermal scopes remain distinct. The old full-length
  fit is not relabelled as the newer width-bucketed fit.

Each per-frame row records saved/ignored/active pulses and distinct occupied
layers by sector type/readout side. The geometry-qualified candidate-tuple count
is only the existing conservative resource bound, not a passing-coincidence
count. No PE/time/slope decision, native membership or temporal coverage is
computed. Missing a saved coincidence control remains explicit.

An identity-compatible table is eligible for a separately reviewed finder-only
check, not automatically an informative nonempty control. No new generation or
full campaign follows from this inventory. F and finite-library uncertainty stay
null; C4 remains stock.

## Locally retained three-event controls

`veto.boundary_pulse_inventory` separately inventories the existing three-event
boundary/downstream controls and their already-completed stock-ADC refits. It
reads an existing laptop copy; no transfer, refit, response or finder is performed.
Externally pin the downstream summary, its final saved report, and boundary
summary; match every consumed local payload to the recorded hash. Original remote
raw/source verification is inherited from that specific final report, not rerun
or replaced by local path substitution.

The old pulse tables store `wf` but not `starttdc`; recover it only through their
actual saved waveform table, verifying event/channel and bijection. The old
boundary event summary omits run/subrun; recover these only from the bound saved
stock-pulse collection, requiring exactly one full identity for every target,
including any quiet frame. Missing or ambiguous targets fail rather than guessing
a run number. No new frame IDs are assigned:
the full 9,000-row registry is not part of this local inspection. Check original
relative time against fitted peak plus saved channel offset, then the exact
forward reco-time shift using the rounded event bunch time. Fit flags/status and
native order are retained. The stock-ADC refit is a distinct pulse population,
not numerically identical stock pulses and not the new width-bucketed fit.

At most two blocks/three frames, 26 pulse tables (24 response model/arm tables
plus two saved refits), 2,048 rows per pulse/waveform table, 512 pulses per frame,
and 20,480 condition rows; use the same metadata byte caps. Output is one exclusive
JSON file at most 2 MiB, including source snapshots, with no atomic-completion or
crash-durability claim. No dummy full-registry identifier is invented.
No all-live-channel, known-nonempty coincidence, native stock-equivalence or
complete-registry admission claim follows. This is another candidate inventory,
not a replacement for the two-control remote inventory.
