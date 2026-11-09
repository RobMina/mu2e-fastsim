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

## Packed sorts (opt-in)

`pack_sort: "packed"` makes the two tracker orderings single-key sorts, which XLA runs as radix sorts on a GPU:
- the per-straw-end cluster order, `lexsort((time, label))` over the clusters and their ghosts, becomes one stable
  sort of a uint64 key, with the label in the high 32 bits and the float32 time's order bits in the low 32 bits;
- the output order, `lexsort((t_analog, sid, event))`, becomes two stable sorts, by time and then by (event, straw).

Times follow `lax.sort`'s float order (-0 equal to +0, NaNs last), so the orders, and hence the digis, are those of
the lexsorts. Labels must fit in 32 bits (batch_events x 131,072 < 2**32). Measured on one batch of 256 mixed events
(A100): tracker 0.36 s -> 0.24 s at an occupancy-only output, 0.91 s -> 0.80 s for full digis.

## Batched charge blocks (opt-in)

`charge_blocks_per_launch: G` runs the drift/charge kernel on G blocks of 4,096 clusters
per launch (vmapped, as the staged `digi_jax` runs `CHARGE_BLOCKS` per call) instead of one
block per sequential step. Every block keeps its key (`fold_in(key, 1)` then its first
cluster offset), so outputs equal the one-block path; `drift_clusters` is padded to a
multiple of `G x 4096`. Without the key the drift is unchanged.

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
