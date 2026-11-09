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

## How they were made

- **Sizes** are 1.25 times the largest demand measured over 47 batches of 128 real mixed events: a development pass, a
  held-out pass and three warm passes. The batch-64 file scales the summed demands by 0.6.
- **Layout choices** were picked by timing sweeps:
  - `clusters_per_step` 64 with long steps gathered to 1,109;
  - `clusters_per_end` 24 with end tiers at 64, 192, 768, 1,536 and 6,144 clusters;
  - 16 charge blocks per launch;
  - scatter CRV waveforms.
- The two widest end tiers keep at least 256 and 16 straws, above their measured demand.
- `run1a_mix1bb_batch256.json` comes from its own census: 42 batches of 256 events, margin 1.25. Its SiPM cross-talk
  candidate buffers `ct1`-`ct3` have no demand counter. They were cut 8x from the generous sizes and checked over
  full development and held-out passes; a further 4x overflowed. As with every buffer, an overflow rejects the batch.

- **Flat cluster sampling** (`flat_clusters`, `minion_steps`, `minion_search: "compare_all"`, `minion_unroll: 4`;
  see [../DEVICE_TRACKER.md](../DEVICE_TRACKER.md)) is sized the same way: 1.25 times the largest demand over full
  development and held-out passes (22 batches of 128 events or 11 of 256 each). The batch-64 file takes 0.6 of the
  batch-128 sizes. These keys change the time, not the digis.

## When a batch does not fit

A batch whose demand exceeds any capacity is rejected as a whole. No output is written for it, nothing is truncated,
and the error names each overflowing buffer. Other pileup intensities, geometries or batch sizes need new sizes:
measure the demand on a representative sample, scale it by a margin and keep the layout. Larger sizes cost memory and
time but never change a physics result. CRV random draws depend on buffer shapes, so a size change gives a
statistically equivalent realization.
