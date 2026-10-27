# Retained measured pulse shapes on frozen pair support

`objective.joint_shapes` reads existing joint captures, completed pair reports and their **original** P4
EventNtuple. It does not run response, reconstruct pulses/clusters, refit lines, change an objective/calibration,
write new pulse captures or match pulses across arms.

```bash
python3 -m crvfastsim.objective.joint_shapes --data SAVED_DATA_CAPTURE \
  --previous PAIRS_A.json --previous PAIRS_B.json --out NEW_REPORT.json
```

`--max-shards 1` audits the first saved shard as a bounded pilot. Its report explicitly has `full_file:false`;
partial cell censuses must be subsets of the historical data, not full-population closure. Use a distinct output
for the full file. The full-file run reproduces the historical data cell sets, pair counts, event/slot counts and
all saved scalar pair moments before interpreting supplemental fields. Raw captures and reports are immutable.

## Identity and integrity

- Require complete unshifted single-source data provenance, the exact previous capture set/SHAs, summary/provenance
  hashes, highest tree cycle, persisted entry census, full ROOT SHA256 before and after reading, and (for the full
  file) the complete ordered event-key SHA and unique integer event census, including empty events.
- Join by `(run, subrun, event, event-local crvpulses index)` in this exact ROOT product. The index was saved before
  selected-pulse filtering. It is not asserted to equal an art collection key or an ordinal in another arm.
- Explicitly verify bar, SiPM, sequence, chosen cluster, peak time and raw area PE, plus clipped analysis PE. Values
  must match exactly after lossless promotion; matching NaNs remain invalid evidence. Missing/duplicate identities,
  out-of-range ordinals and inconsistent branch lengths fail. Never repair with nearby time/PE/channel matching.
- Read shards serially and operate on flat arrays. No per-event Python loop or full pulse-pair Cartesian product.
- Source and selected metadata/capture inputs are hashed before/after. Histories and prior full capture/replay
  gates are inherited evidence, not re-executed simulation. Current source is not certified historical executable
  or cabling/conditions provenance.

## Populations and quantities

The original complete finite, two-projection, selected first-pulse timing groups with full event/slot count >=3
are unchanged. Supplemental nonfinite fields, beta <=0, height PE <=0 or negative pulse height are counted as
shape-invalid. Resulting pair losses are explicit; original group multiplicity is never recomputed after a
shape exclusion. Report the peak and derived-LE quantities on the **same shape-valid retained pairs**. Empty
frozen cells are unsupported, never filled, dropped or reweighted away.

For every body and both saved A/B target supports, preserve body × side × along × path × both sensor PE bins ×
exact/banded N cells and fixed data pair weights. All distinct-sensor combinations on the same bar/end contribute,
including repeated pulses on a channel. Lower-index sensor minus higher-index sensor defines every difference.
Shape moments are computed from per-sensor counts/means/centered covariance, then the fixed 25 PE-bin crossings.
The derived LE second moment is also accumulated directly from centered `t-alpha*beta` combinations to avoid
subtractive cancellation; the expanded covariance identity is an independent check. Both all-selected and
PE-in-range shape-invalid counts are explicit, and summaries distinguish original support coverage from
supplemental shape retention. Shape losses never silently inherit the original pair counts.

Let `dt = peak_low - peak_high`, `dbeta = beta_low - beta_high`, and `alpha = 0.985`. Report
`E[dt²]`, `E[dbeta²]`, `E[dt*dbeta]` and `E[(dt-alpha*dbeta)²]` on identical support. The latter is a
**derived LE proxy**, not an exported threshold-crossing measurement or a commissioned timing estimator:

```
E[dLE²] - E[dt²] = alpha² E[dbeta²] - 2 alpha E[dt dbeta]
```

A smaller derived moment can result from parameter covariance/selection and does not identify an upstream
cause, prove better resolution or justify a correction. Beta is a fitted shape parameter, not arrival spread
or timing error. Area PE divided by height PE and beta is algebraically calibration-dependent; it is not an
independent shape measurement. The stored chi2 is an implementation-dependent reduced fit metric; negative
sentinels and beta==19 counts are explicit but neither reconstructs all fit flags/NDF nor provides chi2 p-values.

## Model-implied calibration, not refitting

The report also evaluates `C_area = e*pulseHeight*pulseBeta/PEs` and
`C_height = pulseHeight/PEsPulseHeight` on finite positive **selected timing members**, not pair-weighted rows.
Under the recorded Gumbel convention these are the effective calibration factors algebraically implied by the
stored pulse fields (including the described fallback formula). Vectorized channel quantiles expose within-channel
spread and channel-median ratios to the frozen MC CSV. Constant values across pulses establish algebraic
consistency, not independent historical DB retrieval or physical calibration validity. These values are **not**
written into a response, used to alter time/PE, or called a measured contribution to the MC–data timing gap.

`python3 -m crvfastsim.objective.joint_shapes_report --report FULL.json --out NEW_DIRECTORY` writes a compact
checked Markdown table and input/source receipt without rereading ROOT. It refuses partial pilot reports.

## Conditional uncertainty

For fixed support cells and fixed weights `w[c]`, use full events (not pairs/slots) as clusters. With pair count
`N[ec]`, summed moment `Q[ec]` and cell mean `M[c]`, event influence is
`psi[e] = sum_c w[c]/N[c] * (Q[ec] - M[c]*N[ec])`. The SE is
`sqrt(G/(G-1) * sum_e psi[e]²)`, where G is the complete analyzed event universe. Combine multiple bars and slots
in one full event before squaring. The LE-minus-peak contrast uses the difference of the two event influences.

These are conditional descriptive SEs assuming independent events. They do not include uncertainty from learning
support/target weights, temporal correlation, shared calibration constants or the frozen MC comparison. P4 belongs
to the scoring reference, not an independent validation holdout. No MC pulse-shape comparison is fabricated when
matching frozen MC upstream products are absent.
