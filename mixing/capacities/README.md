# Device capacities for Run 1A mixing

Explicit buffer sizes for the resident device backend (`--device-capacities`, see
[../DEVICE_BACKEND.md](../DEVICE_BACKEND.md)) with `mixing.device_physics:model`, for the Run 1A
`CeMLeadingLogMix1BB` configuration: a CeMLeadingLog primary plus the MuStopPileup, EleBeamFlash, MuBeamFlash and
NeutralsFlash pools at the stock intensity law.

| File | Batch | Intended device |
|---|---|---|
| `run1a_mix1bb_batch128.json` | 128 events | a full 40-80 GB GPU (electronics groups of 8,192 straws) |
| `run1a_mix1bb_batch64.json` | 64 events | a 24 GB GPU slice (groups of 4,096 straws) |
| `run1a_mix1bb_batch256.json` | 256 events | a 32-80 GB GPU (groups of 8,192 straws); SiPM cross-talk candidate buffers `ct1`-`ct3` at 1/8 of the earlier generous sizes |
| `run1a_mix1bb_batch256_reuse.json` | 256 events | as batch 256, with the opt-in donor-frame digi reuse (an approximation; see [../DEVICE_TRACKER.md](../DEVICE_TRACKER.md)) |

## How they were made

- **Sizes** are 1.25 times the largest demand measured over 47 batches of 128 real mixed events: a development pass, a
  held-out pass and three warm passes. The batch-64 file scales the summed demands by 0.6.
- **Layout choices** were picked by timing sweeps:
  - `clusters_per_step` 64 with long steps gathered to 1,109;
  - `clusters_per_end` 24 with end tiers at 64, 192, 768, 1,536 and 6,144 clusters;
  - 16 charge blocks per launch (128 in the batch-256 files: 12 launches instead of 92 per batch);
  - scatter CRV waveforms.
- The two widest end tiers keep at least 256 and 16 straws, above their measured demand.
- `run1a_mix1bb_batch256.json` comes from its own census: 42 batches of 256 events, margin 1.25. Its SiPM cross-talk
  candidate buffers `ct1`-`ct3` have no demand counter. They were cut 8x from the generous sizes and checked over
  full development and held-out passes; a further 4x overflowed. As with every buffer, an overflow rejects the batch.

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
  time, not the outputs.
- **`flat_saturation_rows: "auto"`**: saturation chunks of one row per straw in float64 (exact there); float32 keeps
  two per straw.
- **`flat_saturation_pool: "tier"`**: saturated ends of all launches of a kernel shape in shared chunks (`"auto"`: in
  float32 only); exact in both dtypes; it has no size.
- **`flat_saturation_tiles: "bounded"`**: with the pool, each saturation chunk runs only the response tiles its ends
  need (exact); it has no size.

- **`electronics_launches: "live"` and `narrow_tier: "compact"`** (batch-256 files; see
  [../DEVICE_TRACKER.md](../DEVICE_TRACKER.md)): the narrow tier's live straws first and launches without a live
  straw skipped. They have no size and change the time, not the digis; with them the largest per-launch
  digitisation demand is 0.93-0.95 entries per straw, inside `digit_entries_per_straw`.

- **`output.tracker_rows`** (batch-256 files): the tracker digi rows downloaded per batch, 229,376: 1.39 times the
  largest demand (164,716 digis in one batch of 256) over full development and held-out passes in float32 and
  float64. The other rows of the padded table are empty; it changes the time, not the outputs.
- **`tracker_reuse`** (`run1a_mix1bb_batch256_reuse.json` only; opt-in approximation): `compute` is the tracker for
  the straws still digitised per batch (primary, two or more sources, the streams not reused), 1.25 times their
  demand over a full development pass; `precompute` sizes the setup batches of 8,192 donor frames (`long_steps`,
  `minion_steps`, `flat_clusters` and `drift_clusters` raised; setup only); `straws` holds the reused straws per
  batch; `streams` lists the reused donor streams.

## When a batch does not fit

A batch whose demand exceeds any capacity is rejected as a whole. No output is written for it, nothing is truncated,
and the error names each overflowing buffer. Other pileup intensities, geometries or batch sizes need new sizes:
measure the demand on a representative sample, scale it by a margin and keep the layout. Larger sizes cost memory and
time but never change a physics result. CRV random draws depend on buffer shapes, so a size change gives a
statistically equivalent realization.
