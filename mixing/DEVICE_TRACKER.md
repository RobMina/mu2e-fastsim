# Resident physical tracker digitisation

`device_tracker.prepare(conditions, capacities, batch_events)` returns numeric
`assets`, `apply(collection, timing, assets, key)` and diagnostic stage functions.
It reuses the unchanged `digi_clusters_jax` sampling, `digi_jax` charge/drift and
`_electronics_kernel` physical implementations. No shared tracker file is
modified. Track reconstruction is not part of this adapter.

## Device operations

- Validate live flat steps and geometry/event identities; apply explicit MC PB
  offset, stock onspill folding and energy/time/read-all selection.
- Existing per-original-step RNG domains produce single and multi-clusters.
  Stable device compaction puts singles first, then multi-clusters in original
  order; minion clusters expand into individual drift electrons on device.
- Drift/charge uses the original 4096-cluster blocks and key folding. All padded
  lanes are masked from output. Tracker computation remains float64 with the
  original StrawCluster float32 charge/time rounding.
- Device grouping identifies event/straw instances. Onspill ghosts retain the
  original, early, late ordering and strict boundaries; shifts happen in float64
  and round back to float32. Charge/wire/end/step provenance follows the copies.
- Device sorting packs full straw ends in time order. Panel jitter retains the
  original 128-event block key domains. The unchanged electronics kernel performs
  threshold search, paired ends, analog/digital dead time, saturation, TDC/TOT,
  ADC and PMP. Final digis are sorted by event/straw/analog time.

No intermediate `np.asarray`, host nonzero/sort/packing or CPU fallback remains in
`apply`. The static chunk loop is `lax.map` inside the compiled graph, not a Python
per-event loop. Output compaction and ROOT writing occur only after the terminal
device download in `device_backend`.

## Conditions as explicit operands

The shared kernels capture device condition arrays in closures. Older CUDA JAX
lowerers download captured constants while compiling an enclosing graph. The first
GPU integration correctly failed the transfer guard at such a copy, before
output. `device_kernels.lift` now traces the original unwrapped kernel at setup and
passes its captured arrays as explicit resident operands to the unchanged JAXPR.
There is no shared-code patch or new physics implementation. Conditions are copied
at setup so later caller mutation cannot change a partially compiled model.

## Capacities and errors

Explicit positive integers:

- `clusters_per_step`: ionisation-cluster slots per step before minion expansion.
- `drift_clusters`: total drift electrons/non-minion clusters in a batch.
- `straws`: event/straw instances; a multiple of `electronics_batch`.
- `clusters_per_end`: per-straw-end slots including onspill ghosts.
- `pairs_per_straw`: threshold-pair/output capacity.
- `electronics_batch`: number of straws passed to the electronics kernel at once.

Every capacity is a resource bound, not a physics cut. Overflow flags reject the
entire combined batch. There is no attempt to substitute NumPy, accept a truncated
tail, or change the physical draw. The electron residual loop remains the existing
uncapped masked loop. Increasing capacities can increase memory/compilation costs;
small fixture sizes are not recommendations for dense production pileup.

The adapter supports the existing conditions' scope: no unimplemented noise hits,
crosstalk, DriftSingleElectrons or UseStatus modes are introduced. Float32 tracker
mode is rejected rather than omitting its relative-frame corrections. CRV precision
is configured independently as in the existing libraries.

## Tiered capacities (opt-in)

Real mixed events are dense and uneven: most steps have a few ionisation clusters and
most straw ends a dozen, but minimum-ionising steps (exempt from `MaxNClusters`) and busy
straws reach hundreds to thousands. One padded width for every lane makes the cost scale
with the busiest one. Two optional capacity groups keep the same physics with less padding:

- `long_steps` + `clusters_per_long_step`: steps with more than `clusters_per_step`
  clusters are gathered into `[long_steps, clusters_per_long_step]`; the others stay in the
  dense `[steps, clusters_per_step]` path. Overflow flags: `long_steps`, `clusters_per_step`.
- `end_tiers: [[straws, width], ...]` with increasing widths above `clusters_per_end`:
  each straw goes to the narrowest tier that holds its busiest end, and each tier runs the
  unchanged electronics kernel at its own width. Overflow flags: `end_tier_K_straws` per
  tier, `clusters_per_end` beyond the widest tier.

Per-slot cluster keys and per-straw electronics keys do not depend on the padded width,
so both modes give the same clusters and digis as one wide capacity (tested exactly in
float64; in float32 see below). Without these keys the adapter is unchanged.

## Flat cluster sampling (opt-in)

The dense path draws every step's `clusters_per_step` slots, including steps with no
clusters and the padding past each step's count, and runs the minimum-ionising electron
loops on every row. Real mixed batches use about 3 M of 16 M dense slots.

- `flat_clusters` + `minion_steps` (set together, with the long-step tier): the short
  steps' clusters are drawn as a flat list of live (step, slot) pairs, at most
  `flat_clusters` of them, and the minimum-ionising loops run on the short
  minimum-ionising steps only, at most `minion_steps`. The per-slot and per-step keys,
  the arithmetic and the cluster order are those of the dense path, so the clusters are
  the same. Overflow flags: `flat_clusters`, `minion_steps`.
- `minion_search: "compare_all"` and `minion_unroll: N`: the minimum-ionising loops'
  truncated-draw search in one comparison pass instead of a device loop per call, and N
  loop bodies per device-loop iteration. Both are `digi_clusters_jax` knobs that leave
  the outputs unchanged (the same indices; extra bodies past a lane's end are identity).

Measured on one real batch of 256 mixed events (A100, occupancy boundary): tracker
0.68 s -> 0.36 s with all three, and digis bit-identical to the dense path in float32
and float64.

## Flat digitisation (opt-in)

The electronics kernel digitises pair by pair: `pairs_per_straw` passes over every straw of a launch, each
evaluating the TOT and ADC voltage sums for every lane, and the saturation response for every lane once any end in
the launch saturates. A real batch has about one threshold-crossing pair per straw, so most of that work is
padding.

- `digit_entries_per_straw: f` (a float in (0, `pairs_per_straw`]): each launch of B straws digitises the flat list
  of its present (straw, pair) entries, at most `ceil(f x B)`, with the pair loop's per-straw, per-stage, per-pair
  keys and arithmetic.
- The digital dead time is applied afterwards, per straw in pair order, as the loop does.
- Saturated ends are compacted into chunks of rows; a chunk with no saturated end is skipped. Three layout choices
  are the defaults with the flat digitisation. Each gives the same digis in both dtypes; the keys are kept so that
  existing capacity files still load, and only the values below are accepted.
- `flat_saturation_rows: "auto"` (default): chunks of B rows in float64 and the loop's own 2 x B rows in float32.
  About 40 % of the 2 x B rows are live in a real batch, so B rows halve the padded saturation work. Over full
  development and held-out passes B rows are bit-identical in float64, but not in float32: the float32 response
  sums depend on the reduction shapes, and B rows changed a few ADC counts. So float32 keeps 2 x B. (The former
  `"lanes"`, 2 x B in float64 too, was removed.)
- `flat_saturation_pool: "tier"` (default) saturates the ends of all launches of one kernel shape (the narrow
  launches, then each end tier) together, in chunks of the same row count. Each chunk computes as a per-launch chunk
  does, so the digis are the same in both dtypes; but a launch's partly filled chunk is filled by the next launch's
  ends, so fewer chunks run. In a real batch of 256, the 34 narrow launches run 26 chunks one launch at a time and 7
  pooled (float32). The launches' saturated-end data stay on the device between the two steps (a few hundred MB at
  batch 256). `"auto"` pools in float32 only and saturates per launch in float64 (before bounded tiles, pooling
  measured 2 % slower there). (The former `"launch"`, per launch in both dtypes, was removed.)
- `flat_saturation_tiles: "bounded"` (default, with the pool) orders the pooled saturated ends by their live response
  tiles and runs each chunk's tile loop only as far as its ends need. A tile past every end's last response step
  holds zeros either way, so the digis are the same. Real saturated ends need about 80 of the 326 response steps
  (90 % need at most 2 of the 6 tiles), so the tile work falls to x0.65 (float32) and x0.51 (float64). With it,
  pooling pays in float64 too. (The former `"all"`, every tile, was removed.)
- The pair loop (no `digit_entries_per_straw`) is the reference for all three: the tests compare the flat
  digitisation with these defaults, and with the `"auto"` pool, to its digis in float32 and float64.
- Overflow flag: `digit_entries_per_straw`. The demand counter `digit_entries_per_straw_milli` is the largest
  per-launch entry count over B, in thousandths (batch metrics are integers).

Measured on real batches of 256 mixed events (A100, full digis): tracker apply 1.28 s -> 0.94 s, digis bit-identical
in float32 and float64. At an occupancy-only output XLA already drops the TOT, ADC and saturation work, so the key
then changes little.

## Packed sorts (opt-in)

`pack_sort: "packed"` makes the two tracker orderings single-key sorts, which XLA runs as radix sorts on a GPU:
- the per-straw-end cluster order, `lexsort((time, label))` over the clusters and their ghosts, becomes one stable
  sort of a uint64 key, with the label in the high 32 bits and the float32 time's order bits in the low 32 bits;
- the output order, `lexsort((t_analog, sid, event))`, becomes two stable sorts, by time and then by (event, straw).

Times follow `lax.sort`'s float order (-0 equal to +0, NaNs last), so the orders, and hence the digis, are those of
the lexsorts. Labels must fit in 32 bits (batch_events x 131,072 < 2**32). Measured on one batch of 256 mixed events
(A100): tracker 0.36 s -> 0.24 s at an occupancy-only output, 0.91 s -> 0.80 s for full digis.

## Electronics launches

Each end tier runs its electronics kernel over the tier's padded straws in launches of `electronics_batch` straws
(a tier smaller than that is one launch). With `narrow_tier: "positions"` the narrow tier's live straws sit at their
positions among all the batch's straws, so a real batch of 256 fills 26 of its 34 launches to about 77 % and leaves
8 empty.

- `electronics_launches: "live"` (the default) skips a launch with no live straw; its rows are empty (mask False).
  The electronics draws are keyed per straw, so the other launches do not change.
- `narrow_tier: "compact"` (the default with `end_tiers`) puts the narrow tier's live straws first, as the end tiers
  already are, so its launches are full and the rest are skipped: 20 launches instead of 26 in that batch. The
  largest per-launch digitisation demand stays at 0.93-0.95 entries per straw, so `digit_entries_per_straw` 1.182 is
  unchanged.
- `electronics_launches: "all"` and `narrow_tier: "positions"` keep the earlier layout as the tests' reference
  (each is one line): the pair loop runs the same launches, so it cannot serve as the reference here.

`"live"` and `"compact"` change the time, not the digis. Over full development and held-out passes at batch 256,
with the device-loop setting on (core `MU2EFS_DEVICE_LOOPS`, the W6 default), the output files are identical to
those with `"all"` and `"positions"` and without device loops, for CRV float32 and float64 with the float64 tracker,
and for the float32 tracker. Measured on one batch of 256 (A100): tracker apply
0.351 -> 0.340 s (float64), 0.366 -> 0.360 s (float32). Launches of 4,096, 16,384, 32,768 or 65,536 straws were
slower, and above 8,192 they change float32 ADC counts (the saturation chunks follow the launch size).

## Bounded electronics loops (opt-in)

`electronics_loops: "bounded"` (with the flat digitisation) changes two fixed-length loops of the electronics kernel:

- the fine threshold-crossing search (regula falsi, at most 10 passes) stops once no lane is still refining; a later
  pass leaves every lane unchanged;
- the saturated ADC samples are summed tile by tile over the response tiles that ran (with
  `flat_saturation_tiles: "bounded"`, about 80 of the 326 response steps) instead of over every step; the steps past
  them are zero.

The sums are in a different order, so equality is up to rounding; on one real batch of 256 the digis were identical
in float32 and float64. Tracker apply on that batch (A100): 0.276 -> 0.262 s (float64 tracker), 0.295 -> 0.282 s
(float32 tracker). Saturated-sample tile widths of 16, 32 and 128 steps and 4, 16 or 32 peak candidates per pass
were also tried: at most 1 % faster, and widths 16 and 128 change float32 ADC counts.

## Batched charge blocks (opt-in)

`charge_blocks_per_launch: G` runs the drift/charge kernel on G blocks of 4,096 clusters
per launch (vmapped, as the staged `digi_jax` runs `CHARGE_BLOCKS` per call) instead of one
block per sequential step. Every block keeps its key (`fold_in(key, 1)` then its first
cluster offset), so outputs equal the one-block path; `drift_clusters` is padded to a
multiple of `G x 4096`. Without the key the drift is unchanged.

The launches run in sequence, and each one's Polya-gain draws (`random.gamma`) loop until every lane is accepted, so
fewer, wider launches pay: on one batch of 256 (A100), tracker apply 0.336 / 0.363 s at G = 16, 0.304 / 0.326 at 32,
0.296 / 0.314 at 64 and 0.292 / 0.310 at 128 (float64 / float32 tracker), digis identical. The batch-256 capacity
files use 128. Skipping launches without a live cluster (a `lax.cond` per launch) was tried and dropped: it changes
how XLA fuses the float64 kernel, which moved 150k cluster times by 1-2 float32 ulps and then some digis.

## Donor-frame digi reuse (approximation, on in the default file)

`mixing.device_reuse`, enabled by a `tracker_reuse` block in the capacity file. The W6 default file
`run1a_mix1bb_batch256.json` has it; `run1a_mix1bb_batch256_exact.json` is the same file without it (and without the
CRV thermal approximation), the exact reference.
About 92 % of the hit straws of a mixed event see steps of one donor frame only.
- **Setup:** every donor frame of the reused streams (MuStopPileup and MuBeamFlash) is digitised once per distinct
  `mc_pbtime` of the signal events (`shifts: "events"`; 5 values 5 ns apart in Run 1A), as an event of its own with
  that time shift. The digital dead time is left out there and applied per event. In the Run 1A pools: 250k frames
  x 5 copies, 6.7 M digis, 70-80 s on one GPU.
- **Per batch:** a straw hit by a single donor frame of a reused stream takes the digis of that frame's copy at the
  event's own `mc_pbtime`, so no digi is moved by the event's time shift and TDC, TOT, ADC waveform and PMP stay
  consistent (a sample with more than 64 distinct shifts is refused). With `event_jitter` the TDC takes the event's
  ROC jitter instead of the setup's (moves the TDC only, by about 0.7 ns rms). The digitisation window and the
  digital dead time (against the event's other digis on that straw) are applied per event. All other straws (the
  primary's, those with two or more sources, and every straw of EleBeamFlash and NeutralsFlash) go through the
  resident tracker with a smaller capacity set (`compute`). They get exactly the tracker's digis, with the same keys.
- **What it freezes:** per donor frame and time shift, the cluster and noise draws. A frame drawn twice gives the
  same digis, so the small EleBeamFlash pool (about 1,200 frames, drawn about 60 times per pass) and NeutralsFlash
  (4 times) are not reused. With them, TDC, ADC and PMP shapes moved 2-8x past the stock floor.
- **Why one copy per event shift.** One copy at the mean shift, moved per event, keeps the ADC sampling phase of the
  setup: the ADC peak index per TDC phase was off by chi2/ndf about 1,800 against stock. A grid of copies around the
  mean (`phases`, `periods`) left a constant 0.17 ns offset between the grid and the discrete event shifts (peak
  index per TDC phase chi2/ndf 5-6). `phases` and `periods` stay available for samples with a continuous spread of
  shifts.
- **Validation** (held-out subrun 1194, three pileup realizations, float32 and float64, against the stock mixed
  sample with two reseeded stock samples as the floor; control: exact digitisation with another tracker seed):
  - ADC peak index per TDC-phase quarter vs stock: chi2/ndf 0.87-0.99 (float32) and 1.09-1.20 (float64), within
    the exact and control arms (0.78-1.11 and 0.86-1.23). The mean peak index vs TDC phase matches exact with no
    shift (chi2 84/80 and 66/80; control 112/80 and 104/80).
  - TDC1-TDC0 vs stock: 0.73-1.21 (float32), 0.95-1.27 (float64) against 0.85-1.27 and 0.90-1.16 for exact and
    control.
  - Per-plane shapes chi2/ndf below 0.7; digi totals equal to exact in the same realization (0.9998-1.0003,
    +-0.0002).
- **Rate** (A100-SXM4-80GB, batch 256, fresh inputs, full output, median of 3 runs on one node): exact file 725.9
  (float32) / 654.2 (float64) ev/s; reuse with one copy 1209.2 / 1052.9; reuse with a copy per event shift (the
  default) 1218.7 (x1.68) / 1051.4 (x1.61). The copies cost setup (warm 112-118 s against 66-68 s for one copy and
  35-37 s exact) and host memory (12.0 GB peak against 8.7 and 5.9 GB), not rate.

## float32

With `TRKFASTSIM_DEVICE_FLOAT=core` and `MU2EFS_DEVICE_FLOAT=float32` the adapter follows the
staged float32 digitiser (`digi_jax`): cluster sampling and the charge kernel see positions
relative to the straw middle, cluster times relative to whole `COMMON_PERIOD`s (restored in
float64 afterwards), and the electronics kernel gets per-straw times relative to
`n_ref * COMMON_PERIOD` with `n_ref` passed in and `t_analog` restored in float64.
`device_kernels.lift` traces the core float32 kernels with their argument cast and creator
scope. Against the staged float32 digitiser, digitised integers are equal and analog times
agree to float32 rounding (the voltage sums run over the padded width, so the summation order
depends on it; same for tiers versus one wide capacity). Tests:
`tests/test_mixing_device_tracker_float32.py`, run in a float32 subprocess by
`test_mixing_device_tracker.py`. Float64 is unchanged.

## Tests

Actual-JAX CPU tests compare physical outputs to `digi_jax.digitize` on identical
synthetic inputs and keys on/off spill, with explicit PB timing and mixed event
lengths. Comparisons cover integer digis/ADCs and analog times, full single/multi
cluster compaction, minion electron expansion, strict ghost boundary/provenance
parity, larger-capacity invariance, 4096-cluster charge-block crossing and 128-event
jitter-block crossing. Empty/poisoned masked lanes, malformed selectors and IDs,
individual-component aliasing, capacities and fractional-ID rejection are tested.
All dispatch tests use a transfer guard and callback-free traced programs.

These are implementation regressions against the existing fast detector, not new
stock-physics closure or a dense-pileup throughput result. See `DEVICE_BACKEND.md` for
the combined runner and `DEVICE_CRV.md` for CRV scope.
