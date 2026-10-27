# Retained upstream timing audit

`python3 -m crvfastsim.objective.upstream_audit` inventories existing products and compares explicitly named
reconstruction settings. It never simulates, refits, calibrates, rescales, changes selection, or writes pulse rows.
Outputs must be new JSON paths. Run this before requesting new diagnostic capture.

```bash
python3 -m crvfastsim.objective.upstream_audit \
  --config retained_expanded_config.txt --frozen frozen_nominal/r3.json \
  --root retained_ntuple.root --capture saved_joint.npz --out new_inventory.json
```

Repeatable `--joint-point /path/to/nominal` (or `atten`, `yield`, `joint`) checks existing frozen response
configuration/canonical-hash/exposure contracts. It does not read raw rows or replace the full identity/replay
gates. Historical receipts missing only the later opposite-side-attachment field retain their original hash and
selection meaning; the joint reader rejects mixing different finder eras across arms.

Optional `--payload-entries 25000` reads the first bounded entries of an EventNtuple's existing pulse shape and
ADC branches. This verifies populated content rather than treating branch presence as enough. Every extracted
body is reported, including explicit empty bodies. The vectorized summaries use **all sequence-zero pulses**,
not the selected line/common-support population from the pair analysis. Quantiles are descriptive; no uncertainty,
MC/data contrast or independent-holdout claim is inferred. This is not a new waveform or response capture.

## Configuration contract

The parser reads the unique balanced `CrvPass1` block from an expanded configuration dump and the unique pulse
producer/DB blocks inside it. Neighboring process histories are excluded. It is not a general FHiCL include or
reference resolver. Missing/ambiguous blocks or required values fail rather than falling back to a template.

Fourteen pulse settings are compared to both the current preset and, when supplied, the frozen resolved
configuration. Report differences without replacing either configuration. The effective frozen phase/offset mode,
waveform settings and explicit assets are shown separately from current defaults. In particular, per-event/zero
v05 overrides must not be mistaken for the generic per-FEB/database preset.

A `defaultBeta` of 19 ns initializes the free-beta fit and supplies fallback values. It does not constrain every
successful pulse to 19 ns. The dark-noise calibration comment referring to 12.6 ns is not a different setting in
the retained P4 reconstruction. Data and MC comparisons use the reconstructed peak time, not an inferred leading
edge. Beta exactly 19 is reported as a value count, **not a measured failure fraction**: the P4 pulse schema lacks
an explicit failure flag, and successful fits can have that value too. Likewise, proximity to beta limits does not
reconstruct all fit-status predicates.

## Evidence limits

- ROOT lookup selects the highest tree cycle and records its actual key and persisted entry count.
- Exact known local/pnfs filenames are opened as byte streams; no namespace glob/stat/list is used.
- ROOT content checksums and parentage are not certified by metadata or a bounded payload read. Source/config/
  capture inputs are hashed; a preexisting full-file checksum is not silently promoted to a new verification.
- Original joint captures lack waveform/fit/arrival fields. The original P4 ntuple can still contain ADCs, fitted
  beta/chi2/height and PE-from-height. This distinction prevents an unnecessary new capture.
- Retained reference dumps may contain charges, waveform voltages, ADCs, pulses and conditions, but another run
  or MC production is not interchangeable with the frozen Thread C input. Charge times are not photon arrivals.
- Channel conditions and actual waveform/clock distributions require their own evidence. A module-wide time
  shift cancels within a fixed same-end pair; channel differences and phase-dependent fit errors need not.
- Source-level template/phase/numerical differences are candidate explanations, not measured contributions to
  the conditional pair-width excess. Keep background, timecal, EX speed and numerical compatibility settings fixed.

Tests exercise process isolation, ambiguity rejection, quoted braces, highest-cycle lookup, frozen receipt
handling, inclusive bounded reading, all-body summaries, nonfinite fields and refusal to overwrite reports.
