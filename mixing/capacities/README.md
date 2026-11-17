# Device capacities for Run 1A mixing

Explicit buffer sizes for the resident device backend (`--device-capacities`, see
[../DEVICE_BACKEND.md](../DEVICE_BACKEND.md)) with `mixing.device_physics:model`, for the Run 1A
`CeMLeadingLogMix1BB` configuration: a CeMLeadingLog primary plus the MuStopPileup, EleBeamFlash, MuBeamFlash and
NeutralsFlash pools at the stock intensity law.

| File | Batch | Intended device |
|---|---|---|
| `run1a_mix1bb_batch128.json` | 128 events | a full 40-80 GB GPU (electronics groups of 8,192 straws) |
| `run1a_mix1bb_batch64.json` | 64 events | a 24 GB GPU slice (groups of 4,096 straws) |
| `run1a_mix1bb_batch256.json` | 256 events | **the W6 default**: a 32-80 GB GPU (groups of 8,192 straws), with the two W6 approximations, donor-frame digi reuse ([../DEVICE_TRACKER.md](../DEVICE_TRACKER.md)) and CRV thermal noise on counters with steps ([../DEVICE_CRV.md](../DEVICE_CRV.md)) |
| `run1a_mix1bb_batch256_exact.json` | 256 events | the exact reference for the default: the same file without either approximation (every straw digitised per batch, thermal noise on every counter, CRV buffers at their earlier sizes) |
| `run1a_mix1bb_batch256_cpu.json` | 256 events | **the W6 default on CPU**: `run1a_mix1bb_batch256.json` with finer tracker end tiers (below) |
| `run1a_mix1bb_batch256_exact_cpu.json` | 256 events | the exact reference on CPU: `run1a_mix1bb_batch256_exact.json` with the same finer end tiers |

## How they were made

- **Sizes** are 1.25 times the largest demand measured over 47 batches of 128 real mixed events: a development pass, a
  held-out pass and three warm passes. The batch-64 file scales the summed demands by 0.6.
- **Layout choices** were picked by timing sweeps:
  - `clusters_per_step` 64 with long steps gathered to 1,109;
  - `clusters_per_end` 24 with end tiers at 64, 192, 768, 1,536 and 6,144 clusters;
  - 16 charge blocks per launch (128 in the batch-256 files: 12 launches instead of 92 per batch);
  - scatter CRV waveforms.
- The two widest end tiers keep at least 256 and 16 straws, above their measured demand.
- The batch-256 files come from their own census: 42 batches of 256 events, margin 1.25. Their SiPM cross-talk
  candidate buffers `ct1`-`ct3` have no demand counter. They were cut 8x from the generous sizes and checked over
  full development and held-out passes; a further 4x overflowed. As with every buffer, an overflow rejects the batch.
- **`crv.coinc_hits` and `crv.coinc_tiers`** (CRV pulses per event and sector type, all four files): 96 in the
  main buffer, then tiers of 16 events x 192 and 4 events x 384 (`mixing/DEVICE_CRV.md`), sized for production
  rather than by the 1.25 margin. The demand is the per-event pulse count of the T1 and T2 sectors (one sector type)
  and it has an exponential tail: over 200,000 mixed events (two 10^5-event runs) the largest was 186 against the
  former single buffer of 187, and one batch of a third run needed 188. A fit above the 99th percentile (86 pulses;
  scale 16 pulses) gives about 1.4 events per batch of 256 above 96, 0.003 above 192, and P(more than 384) about
  1e-10 per event: under 0.01 expected overflows in 10^8 events (about 17 per 10^6 events with the single 187).

- **Flat cluster sampling** (`flat_clusters`, `minion_steps`, `minion_search: "compare_all"`, `minion_unroll: 4`;
  see [../DEVICE_TRACKER.md](../DEVICE_TRACKER.md)) is sized the same way: 1.25 times the largest demand over full
  development and held-out passes (22 batches of 128 events or 11 of 256 each). The batch-64 file takes 0.6 of the
  batch-128 sizes. These keys change the time, not the digis.
- **Flat digitisation** (`digit_entries_per_straw` 1.182) is 1.25 times the largest per-launch demand, 0.945 entries
  per straw, over full development and held-out passes at batch 128 and 256. Launches are 8,192 straws at batch 128
  and 256 and 4,096 at batch 64, so the same fraction is used for all three files.
- **`pack_sort: "packed"`** replaces the tracker's multi-key sorts with single-key sorts in the same order; it has no
  size.

- **CRV `digi_starts: "rows"` and `pulse_scan: "bounded"`** (see [../DEVICE_CRV.md](../DEVICE_CRV.md)) list digi starts
  from the occupied rows and stop the pulse-fit scans at the longest waveform. They have no size and change the
  time, not the outputs. Both are also the code defaults; the files keep them explicit.
- **`flat_saturation_rows: "auto"`**: saturation chunks of one row per straw in float64 (exact there); float32 keeps
  two per straw.
- **`flat_saturation_pool: "tier"`**: saturated ends of all launches of a kernel shape in shared chunks (`"auto"`: in
  float32 only); exact in both dtypes; it has no size.
- **`flat_saturation_tiles: "bounded"`**: with the pool, each saturation chunk runs only the response tiles its ends
  need (exact); it has no size.
- These three are the code defaults with the flat digitisation; the files keep them explicit.

- **`electronics_launches: "live"` and `narrow_tier: "compact"`** (all four files; see
  [../DEVICE_TRACKER.md](../DEVICE_TRACKER.md)): the narrow tier's live straws first and launches without a live
  straw skipped. They have no size and change the time, not the digis; with them the largest per-launch
  digitisation demand is 0.93-0.95 entries per straw, inside `digit_entries_per_straw`. Both are the code defaults
  (`"compact"` with `end_tiers`); the files keep them explicit.

- **CRV `thermal_counters: "hit"`** (`run1a_mix1bb_batch256.json`, the default; an approximation, see
  [../DEVICE_CRV.md](../DEVICE_CRV.md)): thermal noise only on counters with steps; `thermal` 35,332 and
  `charge_clusters` 47,107 are 1.25 times the largest demand over full development and held-out passes in float32 and
  float64; `ct1` and `ct2` halved. The exact file keeps the earlier sizes (`thermal` 565,032, `charge_clusters`
  257,030, `ct1` 196,608, `ct2` 49,152) and the code default `"all"`.
- **`electronics_loops: "bounded"`** (batch-256 files): the fine crossing search stops once converged and the
  saturated ADC samples are summed over the response tiles that ran; no size.

- **`output.tracker_rows`**: the tracker digi rows downloaded per batch. The other rows of the padded table are
  empty; it changes the time, not the outputs. Each is sized over full development and held-out passes in float32
  and float64:
  - batch-256 files: 229,376, 1.39 times the largest demand (164,716 digis in one batch of 256);
  - batch-128 file: 107,813, 1.25 times the largest demand (86,250 digis in one batch of 128; 88 batches);
  - batch-64 file: 57,154, 1.25 times the largest demand (45,723 digis in one batch of 64; 172 batches).
  The batch-64 and batch-128 census ran on CPU (`mixing.device_capacity` with the full padded table as the
  template's bound, so nothing was dropped).
- **`tracker_reuse`** (`run1a_mix1bb_batch256.json`, the default; an approximation): `compute` is the tracker for
  the straws still digitised per batch (primary, two or more sources, the streams not reused), 1.25 times their
  demand over a full development pass; `precompute` sizes the setup batches of 8,192 donor frames (`long_steps`,
  `minion_steps`, `flat_clusters` and `drift_clusters` raised; setup only); `straws` holds the reused straws per
  batch; `streams` lists the reused donor streams; `shifts: "events"` digitises one setup copy per distinct event
  time shift of the signal events (5 in Run 1A), so a reused digi is never moved in time; `event_jitter` gives a
  reused digi its event's ROC jitter.

- **CPU end tiers** (`*_cpu.json`): on a GPU a launch costs little more for padded straws; on one CPU core every
  padded straw costs its full width. The CPU files keep each file's sizes and options and change only the tracker
  `clusters_per_end` (24 -> 8) and `end_tiers`, in the tracker and in the reuse `compute` tracker. `precompute` is
  unchanged, so a saved reuse table made with the GPU default loads with the CPU default.
  - Tiers at 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1,536 and 6,144 clusters per end, sized from
    a census of straws per busiest-end width over 6 batches of 256 events (development seeds, float64; the reuse
    `compute` census in float32 and float64): max(1.25 n, n + 4 sqrt n) straws, rounded up to whole launches of
    8,192 above 8,192, and at least 16 in the widest tier. Held out (12 batches of 256 events per file on other
    seeds, float32 and float64): no overflow, the fullest tier at 0.82 of its size. The largest per-launch
    digitisation demand stayed at 0.93-1.00 entries per straw, inside `digit_entries_per_straw`.
  - Electronics draws are keyed per straw, so the digis do not depend on the tiers: every W6 output array was
    identical to the GPU file's in float32 and float64.
  - Warm rate on one Xeon Gold 6248 core, both files side by side on one node, 3 passes of 256 events:

    | | GPU file ev/s | CPU file ev/s | gain |
    |---|---|---|---|
    | default (saved reuse table), float64 | 6.27 / 6.64 | 7.39 / 8.03 | x1.18 / x1.21 |
    | default (saved reuse table), float32 | 9.23 | 10.24 | x1.11 |
    | exact, float64 | 3.20 / 3.49 | 3.59 / 3.88 | x1.12 / x1.11 |
    | exact, float32 | 4.89 | 5.38 | x1.10 |

    (Two float64 pairs: the second with the files on swapped cores.) Fifteen electronics kernels instead of five add
    80-115 s of compile at setup, none with a filled compile cache.

## When a batch does not fit

A batch whose demand exceeds any capacity is rejected as a whole. No output is written for it, nothing is truncated,
and the error names each overflowing buffer. Other pileup intensities, geometries or batch sizes need new sizes:
measure the demand on a representative sample, scale it by a margin and keep the layout. Larger sizes cost memory and
time but never change a physics result. CRV random draws depend on buffer shapes, so a size change gives a
statistically equivalent realization.
