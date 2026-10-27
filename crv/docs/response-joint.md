# Passive measured-variable joint capture

`crvfastsim.objective.joint` retains the unbinned information missing from the saved-state diagnosis in
`response-diagnosis.md`. It is an opt-in collector, not a response model, objective, correction fit, simulation
runner or acceptance decision. No existing call site is changed. All inputs are measured quantities; truth
branches are ignored. The caller must pass the **same** `c`, `p` and result `o` used in the already-completed
`crvresponse.accumulate_arrays` call, including any pre-applied frozen time corrections.

## Public API

```python
from crvfastsim.objective.joint import JointCapture

collector = JointCapture()
# In the caller's existing input-chunk loop, immediately after accumulation:
# o = crvresponse.accumulate_arrays(c, p, state, window_ns)
collector.add(c, p, o, event_keys, source_index=0)
summary = collector.write("new_joint.npz")
```

`event_keys` is an integer NumPy array of shape `(o["nev"], 3)`, ordered **run, subrun, event**. Keys must fit
nonnegative signed int64; floating-point keys are refused rather than rounded. Even event numbers above `2**53`
remain exact. Keys are never replaced with local entry indices. `source_index` is a caller-assigned nonnegative
integer identifying an input source; callers should record its filename mapping in their external provenance.
Repeated keys are retained, not silently deduplicated or assumed independent. Pairing different samples remains
the caller's responsibility and needs full keys, source provenance and selected identities, not just common seeds.

- `capture(c, p, o, event_keys, source_index=0)` returns a dictionary of owned NumPy columns for one chunk.
- `JointCapture.add(...)` captures owned columns, adds a monotonically increasing `*_chunk_index`, and returns self.
- `JointCapture.columns()` returns concatenated columns, and `summary()` returns JSON-compatible counts.
- `write(path)` exclusively creates the exact path supplied, stores compressed NPZ columns plus a scalar Unicode
  `summary_json`, and returns that summary. Existing paths are refused. Load with `np.load(path, allow_pickle=False)`;
  decode metadata with `json.loads(str(saved["summary_json"]))`. No object arrays or pickle are used.
- `summarize(columns, chunks=1)` summarizes a direct capture as well as collector columns.

An empty typed chunk and events with no pulses/clusters are supported. A collector with no `add` calls raises
`ValueError`: there is no captured schema yet. Capture validates key/chunk lengths and the selected pulse
(event, sector, body, layer) order. It does not hash or fully compare every input value; the caller must not pass
stale fits. It recomputes only deterministic cluster indices and measured algebra, never `fit_arrays`,
`accumulate_arrays`, `objective` or an RNG. It changes neither the arrays/result nor any accumulator. Inputs can
be released or changed after `add`, because captured arrays are owned copies.

Chunk collection is vectorized: flattened Awkward arrays, NumPy indexing, sorting and `bincount`; there is no
per-event/pulse loop or Python-list conversion. The collector buffers all its chunks until writing and performs
a concatenation at write time: it is **not an on-disk streaming store**. Use a fresh collector and uniquely named
NPZ per bounded input shard when memory matters. Preserve source mapping and chunk ordering externally. There is
no implicit flush, background job, sampling or runtime estimate.

## Reading pre-attachment captures

The comparison reader accepts the historical schema-v1 receipt whose finder lacks only
`add_opposite_side_pulses`. It validates a temporary copy with attachment disabled but retains and checks the
original canonical bytes/hash unchanged. All other missing or changed fields still fail. Current-schema captures
must match the current nominal finder setting. This compatibility applies only to reading saved evidence; it
neither changes current reconstruction defaults nor claims historical and current selections are equivalent.

## Tables and identities

Column prefixes identify independent row populations; join them explicitly, never by row position:

| Prefix | Rows | Identities |
|---|---|---|
| `hit_` | Every chosen-cluster sequence-zero pulse, even without a usable line/along coordinate | Full keys, source/chunk, chunk-local `event_index`, slot, body, layer, side, `bar_id`, event-local `pulse_index`, `sipm_id`, `sequence_index`, `chosen_cluster` |
| `event_` | Four rows per input event, including empty/rejected events | Full keys, source/chunk, chunk-local index, slot, chosen event-local cluster index or -1 |
| `candidate_` | Every raw input cluster, before the EX timing cut or largest-PE selection | Full keys, source/chunk, chunk-local index, event-local `cluster_index`, raw `sector_type`, slot, `ex_cluster` |

Slots are EX, T1, T2, aggregate TAG, numbered 0–3. All 16 bodies and both sides are retained in pulse rows;
T10/T11 and the eight individual taggers are not collapsed. `summary_json` includes names. Unknown raw cluster
sector codes get slot -1 and remain in the candidate table and `unknown_slot` summary. Event `body` follows
`crvresponse` exactly: the body of the chosen cluster's largest-**weight** first pulse, -1 when absent. This is a
representative label, not a new per-body timing-group definition. Candidate `body` is the largest-raw-PE linked
pulse's body (all sequences), with `body_available`; clusters without linked pulses have -1. Candidate body is not
an assertion that a rejected candidate has a line fit. Pulse rows carry their actual decoded bodies.

## Pulse quantities and exact spread population

Lengths are mm, time is ns, PE is photoelectrons, and along/path are dimensionless.

- `pe_raw` is the input pulse PE, including negatives/nonfinite values. `pe` is `o["hit_pe"]`, the objective's
  nonnegative-clipped PE. `pe_path = pe/path` reproduces the objective; `pe_raw_path` preserves unclipped PE/path.
- `along_mm` is the measured line coordinate minus the bar centre, and `along = along_mm/half_length`.
  EX/T2 use the x projection and T1/TAG use the z projection. `path = sqrt(1 + sx**2 + sz**2)` is stored even when
  unusable; `okx`, `okz`, `both_lines` and numeric flags distinguish calculation from validity.
- `time` is the measured first-pulse time; `distance` is the along-line distance to its readout end.
  `time_corrected = time - distance/V_FIBER`. Values calculated outside the bar or with an invalid line are
  retained but must not be interpreted as valid corrections; use `corrected_valid`.
- `has_along` / `spread_member` is **exactly** the objective mask: the relevant projection is valid and
  `abs(along) <= 1`. It does **not** require both projections, finite time, positive PE, or a particular body.
- Timing groups are **(event, slot)**, not (event, body), bar, layer or side. `group_count`, `group_mean` and
  `group_rms` use only `spread_member` rows, with the same `bincount` first/second moments and
  `sqrt(max(E[t²] - E[t]², 0))` calculation as the accumulator. Nonfinite member times contaminate the group just
  as they do there; no silent finite-only filtering is introduced. Counts/mean/RMS are repeated on every selected
  row in the group. An empty group has count 0 and NaN mean/RMS. Histogram filling requires count >=
  `MIN_SPREAD_HITS` (currently 3), exposed as `spread_eligible`; this group flag alone does not make an individual
  row a spread member. Event rows are the unduplicated group table for reproducing `tspread` histograms.
- `time_residual` is `time_corrected - group_mean` for members, NaN otherwise. `residual_valid` additionally
  requires a finite residual. Residuals for small groups remain available, but are not objective spread entries.
  The same centered definition is used by `ftp` for EX/T2; all sectors are saved here. No per-module/side offset
  is estimated or applied. `ftp_eligible`/`ftp_inrange` expose the original EX/T2 population separately.
- `pe_along_eligible` requires both projections and `has_along`; `pe_path_eligible` requires both projections.
  `pe_along_inrange` and `pe_path_hist_inrange` additionally apply the objective histogram axis ranges.

## Selection and tails

Event rows retain selection, chosen cluster, line flags/parameters, chosen time/PE/layer/hit counts,
`dt_ex`, group results, selected first-pulse counts, and candidate counts before and after timing/PE selection.
The raw candidate table retains time, PE, layer/hit counts, the four layer PEs, position/time-calculated flag,
cluster x/y, and linked all/first pulse counts. Its `dt_ex` is relative to the **selected EX** reference; it is
NaN if none exists. `timing_match` uses the exact strict `abs(dt_ex) < DT_MATCH` (currently 20 ns); endpoints fail.
EX selection itself has no timing requirement. `eligible` includes known slot, EX availability, positive PE and,
for non-EX sectors, this timing cut. `selected` distinguishes the largest eligible cluster from other candidates.
`timing_rejected` includes non-EX candidates with an EX reference failing the cut, including nonfinite dt;
`timing_outside_window` distinguishes finite tails, and `timing_unavailable` identifies missing/nonfinite timing.
These counts have no added positive-PE requirement unless explicitly named. The candidate table deliberately
contains **no fitted along/path, corrected time, or fitted residual**. Rejected candidates cannot acquire valid
residuals from the selected-event line by implication.

## Numeric flags and summaries

All numeric values, including NaN/inf and out-of-range tails, are retained; no row is dropped for histogram range.
Measured real-valued columns have explicit `*_finite` and `*_nonfinite` flags. Quantities with a documented
histogram range additionally have `*_underflow`, `*_overflow`, `*_inrange`. Endpoints are inclusive for histogram
ranges (NumPy's last bin includes its upper edge), unlike the strict timing-selection window. NaN has neither
underflow nor overflow; infinities are nonfinite and also classified by their signed range comparison.

Pulse ranges use the existing `crvresponse.E`: along [-1,1], path [1,3], PE/PE-path [0,200], group RMS [0,10];
residual range uses `FTP_EDGES` [-20,20]. These residual range flags are provided for every sector but the existing
FTP histogram only covers EX/T2. Event/candidate `dt_ex` flags use [-25,25]; event `dtp_inrange` is an additional
numeric [-40,40] flag, **not** a body-validity/selection or DTP-histogram-membership assertion. Layer counts use
`E["nl"]` [-0.5,4.5], layer PEs use `E["pel"]` [0,400]; `candidate_pe_layer_eligible` requires a selected cluster
and positive layer PE. `candidate_pe_layer_hist_inrange`, `event_nlayers_inrange` and `event_dt_cluster_inrange`
apply the corresponding original populations. Total-cluster PE has no corresponding objective range and has only
finite/nonfinite flags. Unbounded coordinates/times likewise have only finite/nonfinite flags.
Numeric range flags do not establish fit validity;
combine them with the named population flags. Summary boolean counts for vector layer-PE columns count elements,
not clusters.

The JSON-compatible summary provides chunk/event totals, row counts and all boolean flag counts for all tables,
all sectors (including empty ones), every body and unavailable identities. Event summaries also sum the flow
count columns. Group RMS must be counted on event rows, not duplicated pulse rows. A histogram's in-range count
is not a full-population moment or uncertainty estimate. This collector supplies measured correlations and
selection bookkeeping; it does not infer causality, perform paired event audits, restore unrecorded pulses for
rejected candidates, or change the frozen 32-term objective.

## Local synthetic validation

`tests/test_joint_diagnosis.py` checks agreement with accumulated `tspread` histograms, centered grouping across
bodies, full integer keys, selected pulse/cluster identities, timing-cut endpoints and rejected tails,
nonfinite/negative values, single-projection membership, empty chunks/events, owned buffers, no fitting/RNG/
mutation, chunked NPZ roundtrip, and overwrite refusal. These are local report tests on fabricated measured
arrays, not physics simulation or validation on production data.
