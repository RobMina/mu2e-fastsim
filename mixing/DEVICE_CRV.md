# Resident bounded physical CRV onspill adapter

`mixing.device_crv.prepare(geom, cond, cfg, counters, capacities, tabs, pmap)` returns
an adapter with `kind="physics"`, `support=("crv_digis", "crv_reco")`, `metadata`,
numeric-array `assets`, and:

```python
apply(collection, timing, assets, key) -> (tables, errors)
```

This is a physical CRV response adapter, not the synthetic kernel in
`device_synthetic`. It does not select a device, load a source, implement the parent
model factory, or write output. Upload assets once on the parent's selected device
and call `apply` within its compiled batch graph. `prepare` is host setup; `apply`
contains no host compaction, device-to-host transfer, callback, retry, or CPU fallback.
The physical CRV repository and shared kernels are unchanged.

## Input/output contract

- `collection` has `columns`, boolean `mask`, and integer dense `event` vectors,
  exactly like `device_source`. Every buffer has at least one lane; empty input is
  represented by all-false masks.
- Columns are the existing CRV step fields plus **integer `pdg`, joined upstream by
  particle identity**. The adapter does not join particles or reinterpret IDs.
- `timing` has same-length `mc_pbtime`, `reco_pbtime`, `event_length`, and boolean
  `event_mask` vectors. Event rows and full-ID mapping belong to the parent.
- `tabs` maps configured names to existing `photons_jax.DeviceTable` or
  `lookup.LookupTable` objects. `pmap` is the physical optical pixel map. Geometry,
  counters, calibrated conditions, response configuration and template path are
  explicit setup inputs; no substitute physics assets are generated.
- Returned `tables` has `digis`, `pulses`, `clusters`, each a fixed-size numeric
  dictionary containing boolean `mask` and integer `ev`. Digi fields are `bar`,
  `sipm`, `starttdc`, `adc[:,12]`. Pulse fields are `bar`, `sipm`, `seq`, `idx`, `pe`,
  `time`. Cluster fields are `sectortype`, `pe`, `time`, `poscalc`, `x`, `y`, `z`.
  These match the reduced CRV writer's physical payloads; cluster-member tables
  and full pulse-fit diagnostics are not emitted.
- **Every true error flag rejects the entire batch.** `apply` additionally clears
  all output masks on failure. The parent must check flags before writing or
  committing source state; cleared masks alone are not a successful empty result.
- `charges`, `digis`, `pulses`, and `clusters` methods expose resident stages for
  tests/diagnostics. Their errors must likewise be checked; only `apply` clears all
  tables atomically.

## Preserved onspill physics

The endpoint gate is evaluated before optics, using temporary times folded with
**1695 ns**, not the 1675/1700 ns actual event length. NaN endpoint steps are skipped;
infinite endpoints and malformed live payloads are fatal. Original unwrapped
endpoints, including period-crossing intervals, feed optics. Arrivals are folded
only afterward. Physical/status bits 0 and 2 gate the SiPM population; bit 1 is a
reconstruction veto.

Device row preparation follows `response.photons.iter_rows`: local axes, reflector
and sensor mirrors, stock beta, Gaussian/Poisson photon yields, per-0.1-mm-substep
rounding, propagation speed and attenuation factors. It calls the existing
`photons_jax._kernel_a/_kernel_b` optical lookup kernels.

Thermal attempts are generated on-device for **every live physical event/channel**,
including zero-step events. Attempts have Poisson mean `1400*thermal_rate/p_fire`,
uniform times in `[100,1500)`, and uniform sensor pixels rather than the fibre map.
Optical and thermal candidates enter the same `sipm_jax.device_sipm` call, sharing
pixel recovery and cross-talk; dark descendants retain photon index -1. Optical
arrivals earlier than 100 ns remain eligible; arrivals above 1500 ns do not.

Charge grouping and packing use the stock gap and FEB-phase rules. One phase is
shared by an event's channels on a `(ROC,FEB)`. `waveform_jax.synth_dense` is called
with `stock_scan=True`: the inclusive `[200,1500]` start gate precedes accepting a
12-sample digi; accepted tails are not cropped. Consecutive digis are concatenated
on device before `pulsefit_jax.reconstruct_dense`. Pulse times add calibrated DB
correction when enabled and **subtract reconstructed PB time independently**.
Coincidence packing calls the existing `coinc_jax.find_dense` per sector type.

## Explicit capacities

All entries below are required positive Python integers; unknown entries are
rejected. They are static allocation bounds, not physics cuts.

| Entry | Bound |
|---|---|
| `substeps` | Expanded substeps per configured table/reflector/sensor block |
| `photons` | Arrivals per such block, before SiPM status/time filtering |
| `thermal` | Thermal attempts across the whole batch |
| `ct1`, `ct2`, `ct3` | Cross-talk candidates in each existing kernel generation |
| `pixel_short` | Short pixel-stream table length |
| `pixel_long_streams`, `pixel_long_hits` | Long-stream table rows and length |
| `charge_clusters`, `charges_per_cluster` | Waveform packing rows and charges per row |
| `waveform_samples` | Samples per synthesized charge-cluster waveform |
| `digis` | Total digis in batch |
| `reco_waveforms`, `reco_samples` | Concatenated reconstruction waveforms and sample length |
| `pulse_rounds` | Pulse-fit rounds per waveform; a pulse in the last round is fatal |
| `coinc_hits` | Hits per event and sector type |
| `coinc_combinations` | Filtered hit slots per side/layer in the coincidence kernel |

Pulse-output capacity is `reco_waveforms*pulse_rounds`; cluster-output capacity is
`batch_events*sector_types*coinc_hits`. Total SiPM candidate-buffer length is
`optical_blocks*photons + thermal + ct1 + ct2 + ct3`. Coincidence memory can grow as
the fourth power of `coinc_combinations`; charge waveform intermediates scale with
`charge_clusters*charges_per_cluster*waveform_samples`. **The tiny test capacities
are not production recommendations.** No capacity overflow is accepted as a
truncated event. The fixed three-generation SiPM model is inherited from the
existing physical JAX kernel and recorded in metadata; this adapter does not
claim an unbounded cross-talk cascade or new stock closure.

## Scatter waveform synthesis (opt-in)

`"waveform": "scatter"` in the CRV capacity object replaces the dense
`[charge_clusters, charges_per_cluster, waveform_samples]` gather of `synth_dense` with a
flat scatter-add: each charge adds its template to the few samples it spans. Sample times,
template indices, noise draws (same key and shape) and the gated digi scan are computed as
in the dense kernel; only the summation order of overlapping charges differs. Digis are
tested equal to the dense path, with and without noise. `charges_per_cluster` is unused in
this mode. Default `"dense"` is unchanged.

## Bounded digi listing and pulse scans (opt-in)

Two CRV capacity options restructure padded scans without changing results:
- `"digi_starts": "rows"`: digi starts are listed from the rows of the [charge_clusters, waveform_samples] start table
  that hold one, instead of a `nonzero` over the whole table. The scatter synthesis' gated scan stops at the longest
  waveform. The (row, sample) list is the same, fill values included.
- `"pulse_scan": "bounded"`: the `pulsefit_jax` peak scans stop at the longest reco waveform. The fit rounds stop once
  no waveform found a pulse; a waveform that finds none never finds one later. The found pulses are the same.

On an A100 at batch 256 the CRV stage goes 0.183 -> 0.102 s (float32) and 0.227 -> 0.133 s (float64). Full-pass
outputs are identical to the defaults (dev and held-out, both dtypes). The Run 1A capacity files set both.

## Thermal noise on counters with steps (opt-in approximation)

`"thermal_counters": "hit"` draws the SiPM thermal noise (dark counts) only on the four SiPMs of each counter that
has an in-window step in the event; the default `"all"` draws it on every live SiPM. In real mixed batches of 256
(Run 1A pileup) counters without a step carried essentially no output (0 digis and 0 pulses in two batches, one digi in
a third), but their dark counts were about 94 % of the thermal hits: with the option the thermal hits fall from 451k
to 27k per batch and the charge clusters from 203k to 36k. The batch-256 capacity files set it, with `thermal` and
`charge_clusters` sized to 1.25 times the largest demand over full development and held-out passes (28,265 and 37,685)
and `ct1`/`ct2` halved (full passes run without overflow).

The draws change, so the outputs are statistically, not bit-for-bit, those of the default. Held-out subrun 1194, three
pileup seeds, float32 and float64: against the stock sample the CRV pulse totals and per-sector shapes agree as the
default's do (pulse ratio 0.981-1.004 +- 0.017, sector chi2 <= 1.3/2); against the default at the same seed the pulse
total is 0.06-0.10 % lower on average (per seed -0.19 % to +0.05 %, paired error 0.09 %), and the pulse PE, time and
per-event shapes are within the default's seed-to-seed spread. A real batch of 256 (A100): CRV 0.103 -> 0.093 s
(float32), 0.137 -> 0.121 s (float64).

## Scope and validation

Supported: staged stock onspill ZS response through reduced coincidence outputs.
Unsupported: traps, NZS, empirical background, fused response, automatic capacity
retry, host fallback. Nonstock onspill windows are explicitly rejected.
`JAX_ENABLE_X64=1` is required; existing `MU2EFS_DEVICE_FLOAT` float32/float64 kernel
precision is respected. RNG is JAX-law sampling, not NumPy or art byte replay;
changing shapes/capacities changes the realization.

Local actual-JAX tests exercise transfer-guard dispatch, callback-free tracing,
repeatability, partial batches, empty-event thermal charges, physical readout and
status, uniform dark pixels, shared optical/dark recovery, original optical
intervals and post-optics folding, period/event-length independence, reflector and
propagation row parity, gated waveforms against independent NumPy synthesis,
contiguous-digi pulse fitting, independent reconstructed PB correction,
coincidences against the independent NumPy finder, unsupported configurations,
malformed inputs, and capacity failures. A smoke test uses the available actual v6
`LookupTable_6000_0`, physical `photonMap.root`, and stock single-PE waveform asset.
Small analytic lookup tables in the other tests exercise the actual optical
kernels, not synthetic substitutes in the implementation.

Run from the umbrella with its CRV/tracker/core worktrees on `PYTHONPATH` and
the public Offline waveform on `MU2E_SEARCH_PATH`. Set `CRVFS_LOOKUP_DIR` to an
existing directory containing the v6 table and photon map to include the optional
real-optics smoke; missing optional assets skip without downloading:

```bash
JAX_ENABLE_X64=1 MU2EFS_DEVICE_FLOAT=float64 \
  python3 -m pytest -q tests/test_mixing_device_crv.py
```

Local result: **17/17 adapter tests passed in float64 and float32**, including the
actual v6 asset test; the existing `tests/test_mixing_device.py` also passed 17/17.
The local tests use CPU JAX explicitly available in the environment. They do not
certify GPU execution, representative dense-pileup memory behavior, throughput or
dense mixed production closure. Shared tracker files are unchanged.

Source contracts: `crvfastsim/onspill.py`, `docs/onspill-response.md`,
`response/photons.py:iter_rows`, `response/photons_jax.py:_kernel_a/_kernel_b`,
`response/sipm_jax.py:device_sipm`, `reco/waveform.py:charge_clusters`,
`reco/waveform_jax.py:synth_dense`, `reco/pulsefit.py:concat_waveforms`,
`reco/pulsefit_jax.py:reconstruct_dense`, `reco/coinc_jax.py:pack/find_dense`.

The combined physical adapter subsequently ran on a real RTX A6000
and completed successfully with 95 CRV digis, 95 pulses and 6 clusters
in seven persisted events including warm-up. That integration uses explicitly
synthetic source/CRV calibration data, unlike the local actual-v6 optical smoke;
it is an execution-path test, not production closure. Details: `DEVICE_BACKEND.md`.
