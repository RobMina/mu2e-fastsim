# Saved CE source/stored-PBT composition diagnostic

## Admission and scope

This is the implementation follow-up to the frozen design in
[veto-ce-calibration-validation.md](veto-ce-calibration-validation.md), SHA-256
`ca69360f65abea85e12381601c299850636903fb57358804dd610711667655a1`.
That document records the pre-execution design; this page records implementation
and execution status. The separate calibration, external validation and uncertainty
gates are unchanged. All eleven viewed sources remain development material.

The diagnostic reuses the saved proxy rows, selected-track Parquet and old exposure
ledger to form a new source-by-proton-count-by-MC-PBI-by-stored-PBT table. It does
not re-extract ROOT data, regenerate prior joint/summary products, rerun selection,
decode selected `coincs`, fit calibration/weights or draw any detector response.
Only low-level saved-input validation/join/binning helpers are reused from frozen
`ce_joint_diagnostic.py`, SHA-256
`1d9c4dafc450df68c72d365c82fd46116090a807838a0b07fdb055a410eab4b4`.

## Frozen calculation

Preserve every original source index, including the gap at index 7; all original
count/PBI/time bins, flow and empty cells; all zero-track events; and full event
identities with frozen track-list positions. Strict 85 < p_mid < 200 is the only
window applied to already selected tracks. The complete grid includes other
proton counts and middle MC-PBI bins even though the contrasts concern counts
zero and one at the original lowest/highest MC-PBI quintiles.

For each of those two counts, blocks are source index crossed with stored-PBT bin.
Common support requires N_low >= 100 and N_high >= 100. No selected-outcome
threshold enters this rule. Report three distinct differences in percentage points:

1. **D_all:** endpoint event-selection fractions in all retained endpoint events.
2. **D_common:** endpoint fractions restricted to common support, with each endpoint
   retaining its own composition.
3. **D_standardized:** sum of block selection-fraction differences weighted by
   q=(N_low+N_high)/sum_common(N_low+N_high).

Each count has its own support and q. These are descriptive retained-composition
weights, not CE target corrections. No common support means null, not zero. All
zero-denominator ratios are null. Keep total/supported/excluded N/A/T and their
endpoint-specific fractions, every cell and exclusion reason, per-source contrasts,
shape-support flags, multiplicities and selected-time histograms. Shape-support
flags A>=20 and T>=30 do not choose event-rate support or q.

The prior completed joint report supplies immutable count, matched-track identity,
source and marginal checks; the completed summary supplies immutable raw count and
rate checks. Input pins, tuple joins, strict-window counts, source bindings or
conservation mismatches fail closed. No substitution, deduplication, bin merging,
threshold lowering, pseudocounts or dropping failed cells is allowed.

## Bounded saved-input interface

The new command requires four explicit report paths, all hard-pinned to the
successful products listed in the design contract:

```bash
python3 -m crvfastsim.veto.ce_composition_diagnostic \
  "$ORIGINAL_SELECTION_REPORT" "$ROWS_REPORT" \
  "$COMPLETED_JOINT_REPORT" "$COMPLETED_SUMMARY_REPORT" \
  --outdir "$NEW_OUTPUT_DIRECTORY"
```

A mutually exclusive `--out NEW_JSON` option uses an existing distinct parent.
No paths listed in the completed joint or summary report are opened. The original
selection and rows reports bind the allowlisted saved payloads. Reports are capped
at 8 MiB, source files at 64 KiB, numeric NPZ at 16 MiB, selected Parquet at 64 MiB,
ledger at 8 MiB, and the new output at 2 MiB. Original footer, selected-column and
row/track bounds remain in force. Input and source hashes are checked before use
and again before exclusive report-last publication. A failed write may leave an
incomplete directory/file; there is no atomicity or crash-durability claim. Byte
caps are not whole-process RSS or malicious-decoder allocation guarantees.

Local and remote synthetic tests precede a single admitted real run. Remote
execution needs a fresh idle-host/ownership check, exclusive reservation, healthy
data mount, source-hash agreement and sufficient data quota. Old successful
consumers, fixed stock, inputs and immutable receipts are not overwritten.

## Interpretation limits

This is a **source/stored-PBT composition-standardized retained association**.
MC-PBI is truth diagnostic only; stored PBT is an MC truth-smeared surrogate, not
an independent detector clock. Source files are not physical ancestry units.
Coarse bins leave within-bin composition unresolved. Selection and inclusive
signal contributions remain part of the retained population.

A contrast that shrinks or persists establishes neither causality nor sufficiency,
factorization, calibration or transport to another target. Common support is not
a precision certificate, and an average can hide opposing cell contrasts. Selected
track-time distributions remain conditional on selection. No holdout, significance,
inferential interval, finite-library error or new F follows. Calibration, target
weights, F and uncertainty outputs remain null; C4 stays stock.

## Execution status

Implementation is complete with **98 synthetic tests passing locally (1.00 s)
and remotely (4.42 s)**. Source SHA-256 is
`6d44c2ad88fb94393e98bdb7b2d1ddd61ee45fee854d6bfd41613df69428f858`;
test SHA-256 is
`75ae7e7eabad7c440e1bdc81fbbdfc71f487de416ecdd1097d1dcb820ee4eb9f`.
Exact local/remote hashes agree and the frozen helper/design hashes are unchanged.
The tests include support/composition separation, N-only support even with zero
selected events, count-specific weights, missing endpoints, cancellation visible
by source, flows, multiple equal-valued tracks, large full IDs, source-index gaps,
input mutations, inherited-count corruption, bounds and output races.

## Completed single saved-input run

The run succeeds once: **6.68 s wall, 4.30 s user, 0.42 s system, 491,764 KiB
peak RSS**. The 510,176-byte report has SHA-256
`b4a21d7323cde3c5492765f8df1d2563320802740080b45e51aa2f3e9c696475`.
Runtime NumPy is 2.5.2 and PyArrow 18.0.0. No ROOT or selected `coincs` payload
is opened. The prior joint report is read from its immutable archive copy with
the same frozen hash, so the new output remains outside every input directory.

All count, full-tuple/list-position digest, source and raw-marginal checks pass.
The complete source × proton-count × MC-PBI × stored-PBT grid has shape
**11 × 8 × 7 × 7**, with all flows/empty cells. Totals remain 91,226 retained
events, 42,319 selected events/tracks and 48,907 zero-window-track events. The
1,887,102 global window-selected events outside the retained subset remain
explicitly accounted. The original endpoint contrasts are reproduced exactly.

All following differences are lowest-minus-highest MC-PBI-quintile event-selection
fractions, in **percentage points**, not changes in F:

| Raw nProtonTCs | D_all | D_common | D_standardized | Common source/PBT cells |
|---|---:|---:|---:|---:|
| 0 | 4.041541 | 4.263304 | 4.246582 | 20 of 77 |
| 1 | 4.499489 | 4.375971 | 4.337321 | 24 of 77 |

The common-support sets differ by count. Their pooled endpoint N are 10,133 and
9,071. Relative to D_common, composition weighting changes the contrasts by about
-0.017 and -0.039 points; this comparison does not conflate restriction with
standardization. D_all is not a calibrated target and q is not a correction.

### Explicit support losses

A=T in this particular saved selection; no multiplicity was discarded.

| Count | Endpoint | Total N / A | Supported N / A | Excluded N / A | Excluded N fraction | Excluded A=T fraction |
|---|---|---:|---:|---:|---:|---:|
| 0 | low | 11,035 / 5,339 | 7,931 / 3,839 | 3,104 / 1,500 | 28.129% | 28.095% |
| 0 | high | 3,004 / 1,332 | 2,202 / 972 | 802 / 360 | 26.698% | 27.027% |
| 1 | low | 5,525 / 2,684 | 4,674 / 2,271 | 851 / 413 | 15.403% | 15.387% |
| 1 | high | 5,270 / 2,323 | 4,397 / 1,944 | 873 / 379 | 16.565% | 16.315% |

All excluded cells and their outcomes remain in the report; no adaptive support
rescue is attempted. Count-zero source index 5 has no common-support cell, so its
restricted/standardized source contrasts are null, not zero. Per-source
standardized contrasts range from -1.577 to 8.771 points at count zero (among ten
supported sources), and -0.392 to 10.502 points at count one (all eleven). These
are heterogeneous descriptive associations, not independent-source error estimates.

### Decision

The observed endpoint association persists under this frozen coarse source/PBT
composition weighting on the supported populations. It is not evidence that
source/clock effects have been causally removed, nor proof of an intensity effect.
Support losses, within-bin differences, trigger conditioning and signal contribution
remain. No significance, calibrated weights, uncertainty estimate or new F is
claimed; a one-dimensional proton-count correction is still not admitted.

Reuse this report rather than re-running the successful diagnostic. Further CE
calibration requires the independent reference, historical deployment/IOV/selection
bindings and genuinely unexamined validation population specified in the design.
Neither more tuning of these viewed bins nor a new random split supplies that
evidence. Physical-library reuse and full-timing/persistence remain separate gates.
