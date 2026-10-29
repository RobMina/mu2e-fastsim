# Bounded keyed CE joint diagnostic: frozen contract

## Scope

This is a separately authorized follow-up to the completed scalar-availability
check, not another availability audit, reconstructed selection or response run.
`veto-ce-proxy-semantics.md` defines the raw observables and evidence limits.
None is admitted as calibrated beam intensity. In particular, the named MC
recipe fills reconstructed ProtonBunchTime from smeared truth. The analysis keeps
raw stored time coordinates and never subtracts bunch time again or wraps them.

## Data stages and pins

1. `ce_proxy_rows` reads the exact eleven retained files' IDs and nine scalar
   observables, using the frozen bounded XRootD/metadata helpers. It compares
   every consumed array's dtype, shape and SHA-256 to the successful availability
   report. It persists numeric keyed rows with the source basename index, not
   reconstructed tracks, CRV data, truth arrays or an event-time alias.
2. `ce_joint_diagnostic` reads that small persisted product, the frozen selected-
   track Parquet columns `run/subrun/event/t_mid/p_mid`, and the old seven-column
   exposure ledger. It never opens ROOT files or decodes `coincs`.

Mandatory original selection-report SHA-256:
`f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3`.
Mandatory successful availability-report SHA-256:
`2b59602667126391fe1b32bb6ce4051b83c88268f5f60c580922d1cfeef18325`.
The extractor pins the frozen audit helper source to
`3765b8114f4128d7bae438d3b943cc6f8b61c1c4f393e0963fb4ede6494cc776`.
The joint diagnostic requires an explicit external SHA-256 for the new rows
report and checks its rows artifact binding before reading it.

The original report binds selected-track SHA-256
`7f60b88899f35bea7cf0a438aa18250ff0021ad229bc3c000bfd180fa386a2ba`
and exposure-ledger SHA-256
`b9f94c948099cdeddb1e27634bb4f2f232619aa11d854ae5a96d4d5339e955cf`.
No alternative input, alias, deduplication or source replacement is admitted.

## Identity, selection and denominator

- Full run/subrun/event tuples, each admitted as a nonnegative signed int64
  (larger uint64 values fail, never wrap); no packed keys. Unique event rows in
  each input checked before momentum filtering. Jagged times/momenta must align,
  be nonnull and finite. Equal-valued tracks remain distinct frozen-list entries.
- Apply only the saved strict `85 < p_mid < 200` window; do not rerun cuts or
  choose a best track. Preserve multiplicity and count endpoint exclusions.
- Proxy and exposure-ledger key sets must match completely; stored measured
  nprotons and bunch time must agree exactly between the two products.
- Selection extends beyond these eleven files. Report matched and unmatched
  selected events/tracks separately; no silent anti-join removal from accounting.
  The global window counts and matched counts must reconcile to the original
  selection report. This is an admission check, not an adjustable target.
- Retain zero-selected-track ledger events. Each proxy row's source index maps
  to the actual pinned input, not inferred from subrun or generated ancestry.

For a bin, report separately all retained events `N`, events with at least one
selected track `A`, and selected tracks `T`. The three ratios are `A/N`, `T/N`
and `T/A`; only the first is an event-selection probability. Tracks/event can
exceed one. No selection probability versus track time is defined for unselected
events, because those have no selected-track time.

All probabilities are conditional on retained triggered/triggerable outputs.
They are not generated-CE efficiencies or physical-exposure corrections.

## Binning frozen before execution

| Quantity | Edges / interpretation |
|---|---|
| Stored reconstructed bunch time | `[-240,-230,-220,-210,-200,-190]` ns |
| Raw stored `t_mid` | 0 through 2000 ns in 100-ns increments |
| Raw `nProtonTCs` | `[-0.5,0.5,1.5,2.5,3.5,5.5,9.5]`: categories 0,1,2,3,4–5,6–9,then ≥10 |
| Each hit count and integer caloEnergy | `[0,1,32,64,128,256,512,1024,2048,4096]` |
| MC PBI, truth diagnostic only | Original report's five all-event quantile strata; exact saved edges, not refitted |

All grids retain explicit underflow/overflow. Bins are left-closed/right-open,
except the original final MC-PBI upper edge remains inclusive. No adaptive merging,
pseudocounts or extrapolation. Constants (`measured_nprotons`, `pbterr`, CAPHRI
validity) are recorded as constants, not promoted into fitted predictors.

For each proxy × stored bunch-time grid, and MC-PBI × stored bunch-time grid,
record `N/A/T`, multiplicities 0/1/2/3/≥4, and selected-track raw-time counts.
An additional proton-count × MC-PBI × bunch-time grid is explicitly truth-labelled.
Retain per-source-file counts and timing ranges to expose source/clock-phase
confounding. Histogram totals must reconcile with raw counts.

Support flags are fixed: event denominator `N>=100`; track shape `A>=20` and
`T>=30`. Unsupported cells stay in the report, along with their event/track
fractions. These flags do not establish precision or independence. No inferential
error bars or finite-library covariance are claimed.

## Resource and publication contract

Existing ROOT caps remain: 128 MiB/file, 1 GiB total cumulative reads, 32 MiB/read,
1 MiB/scalar branch, 16 MiB selected scalar payload/file; client timeouts 30 s per
operation, one file at a time. Numeric keyed NPZ output is at most 16 MiB, report
at most 2 MiB. Files are exclusive in a new output directory, with the report
written last; a failed attempt may leave an incomplete directory, not a successful
receipt. There is no whole-process RSS, atomic-publication or crash-durability claim.

Joint analysis uses local-file admission: selection file at most 64 MiB, ledger
8 MiB, reports 8 MiB, bounded Parquet footer/metadata and decoded selected columns
256 MiB; at most 2.1 million selected rows/tracks, 64 tracks/row and 100,000 ledger
events. NPZ membership, headers, numeric types and decompressed sizes are bounded;
no pickle. Inputs/source are rechecked before exclusive final output. Processing
is vectorized over event/track arrays, with loops only over fixed grids, files or
Parquet batches; no event-by-event or track-by-track Python traversal. Declared
selected-column sizes are checked before decoding; actual batch bytes are checked
after materialization. These are not hard malicious-decoder allocation limits.
The joint JSON output has a separate 16-MiB cap.

## Allowed conclusion

The result can reveal dependence and sparse overlap in this retained sample.
It cannot establish independent detector timing, a sufficient intensity proxy,
physical ancestor independence, a calibrated target law, or a replacement for
stock C4. `calibration`, `weight_model`, `F` and `finite_library_uncertainty` remain
null. Successful products must be reused rather than re-extracted for new plots.

## Completed extraction and joint accounting

The slim extraction succeeds once on all 91,226 entries, with every original
consumed-array digest/dtype/shape matching the successful availability audit.
It reads 3,464,527 bytes including header rechecks, without rereading the version
payload. Measured resources: **10.16 s wall, 141,640 KiB peak RSS**. The numeric
rows product is 5,568,104 bytes and its report 92,817 bytes. The extractor's
120 targeted tests pass locally (11.96 s) and remotely (31.08 s).

- Rows report SHA-256:
  `af675694a4a997295d42d84a98bc8fb0aedac160aa83c2cc69002e3cb4e2b2e8`.
- Numeric rows SHA-256:
  `20f012be65a9487226923f0388a72f06eca8ee02c95bcdcadc822b8b68f45f3a`.

The joint diagnostic then succeeds once using **only saved products**. Resources:
**5.66 s wall, 481,408 KiB peak RSS**. The report is 138,519 bytes, SHA-256
`09c8831caca851de8ebb78cee91f467d9ff9f7c6b346ebdffe7ca5a2c70e3831`.
After review added the observed bunch-time-error summary, **103 tests pass locally
(0.78 s) and remotely (2.25 s)**. No measured outcome or input pin was changed.

| Accounting | Count |
|---|---:|
| All frozen selected-file event rows / tracks | 1,930,163 / 1,930,163 |
| Strict-window event rows / tracks, global | 1,929,421 / 1,929,421 |
| Tracks below 85 MeV/c removed | 742 |
| Tracks exactly at 85 or 200, or above 200 | 0 |
| Retained proxy/ledger events | 91,226 |
| Matched selected rows before strict window | 42,336 |
| Matched selected events / tracks after window | 42,319 / 42,319 |
| Retained events without a window-selected track | 48,907 |
| Global window-selected events outside this retained subset | 1,887,102 |
| Proxy/ledger anti-joins, both directions | 0 |

All inherited global, matched and MC-stratum counts reconcile exactly. Multiple
tracks were not silently removed: this particular frozen selection has one track
per selected event. Stored nprotons is constant zero, stored pbterr is constant
0.5 ns, and CAPHRI validity is true in every retained row. These observations do
not establish detector calibration.

Fixed-support failures remain in the saved tables. For proton count × stored
bunch time, 138 events / 67 selected tracks are outside joint support; adding MC
PBI strata increases that to 1,335 events / 621 tracks (1.4634% / 1.4674%). No sparse
cells were dropped or merged to improve a result. All seven grids retain raw
selected-track time histograms and explicit under/overflow counts.

## Saved-table reduction: observations, not a proxy calibration

`ce_joint_summary` consumes only the hard-pinned successful joint JSON, checks
cross-grid marginal/count conservation, and reports raw rates and selected-time
histogram total variation. It never opens paths listed inside that report. Its
52 tests pass locally (0.25 s) and remotely (0.75 s). One real summary took
0.45 s wall / 88,660 KiB peak RSS; output 90,668 bytes, SHA-256
`63fc77886c3febb00ba6120ee9dfd6140bb3fedf12260312544f3bb574b68e0a`.

The overall retained-event selection rate is **46.3892%**, not generated-CE
acceptance. Raw proton-candidate marginals are:

| nProtonTCs | Retained events | Selected events/tracks | Selection rate |
|---|---:|---:|---:|
| 0 | 35,037 | 16,316 | 46.568% |
| 1 | 30,552 | 14,187 | 46.436% |
| 2 | 15,601 | 7,184 | 46.048% |
| 3 | 6,466 | 2,988 | 46.211% |
| 4–5 | 3,173 | 1,461 | 46.045% |
| 6–9 | 392 | 181 | 46.173% |
| ≥10 | 5 | 2 | 40.000%; unsupported |

The nearly flat common-bin marginal does not establish sufficiency. After
summing over the frozen stored-bunch-time bins, the **truth diagnostic** within
zero proton candidates is 5,339/11,035 = **48.382%** in the lowest MC-PBI quintile
and 1,332/3,004 = **44.341%** in the highest. Within one candidate it is
2,684/5,525 = **48.579%** versus 2,323/5,270 = **44.080%**. Thus the observed endpoint
spreads are 4.042 and 4.499 percentage points even at fixed raw proxy count.
These are raw conditional proportions, with no significance/causality statement,
uncertainty estimate or proof of a calibrated residual response. They do **not**
justify replacing MC-PBI dependence with a one-dimensional nProtonTCs weighting.

For the 0/1/2 candidate marginals, selected raw-time histogram TV relative to the
all-selected marginal is respectively **0.00685 / 0.00827 / 0.01082**, using fixed
100-ns bins and including flow. This is a coarse descriptive shape comparison,
not a factorisation or independence certificate. All observed selected-track
counts fall in the 600–1700 ns bins; no extra time cut was imposed by this code.
The 6–9 candidate category has only one supported cell after MC-PBI stratification,
so its reported zero supported-cell rate range must not be called invariance.

Per-file selection rates span 45.343–46.810%; stored bunch-time ranges also vary
by file. The retained mixture of clock phases, source files, intensity and trigger
conditioning is not disentangled by these summaries. Inclusive tracker/calo
activity cannot be treated as an independent beam monitor without further
validation of signal contributions and the response/selection model.

**Decision:** candidate raw observables and their retained joint tables are now
available for reuse, but a calibrated measured-variable target law is **not
established**. Do not apply a mean-ratio F rescaling, bootstrap filenames as
independent physical histories, or fit a deployable correction using truth as an
input. C4 stays stock. Any subsequent calibration/sufficiency study needs a frozen
validation split, justified uncertainty units and historical configuration/IOV
bindings; existing rows and joint tables should be reused.
