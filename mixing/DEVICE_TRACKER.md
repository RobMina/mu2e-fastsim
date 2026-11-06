# Resident physical tracker digitisation

`device_tracker.prepare(conditions, capacities, batch_events)` returns numeric
`assets`, `apply(collection, timing, assets, key)` and diagnostic stage functions.
It reuses the unchanged `digi_clusters_jax` sampling, `digi_jax` charge/drift and
`_electronics_kernel` physical implementations. No shared tracker/TRK2 file is
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
small fixture sizes are not recommendations for dense production PU1.

The adapter supports the existing conditions' scope: no unimplemented noise hits,
crosstalk, DriftSingleElectrons or UseStatus modes are introduced. Float32 tracker
mode is rejected rather than omitting its relative-frame corrections. CRV precision
is configured independently as in the existing libraries.

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
stock-physics closure or a dense PU1 throughput result. See `DEVICE_BACKEND.md` for
the combined runner and `DEVICE_CRV.md` for CRV scope.
