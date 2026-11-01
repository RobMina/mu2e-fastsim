# Device precision (float32 default)

The CRV device chain runs in float32 by default. Set `MU2EFS_DEVICE_FLOAT=float64` to get float64, whose output is
identical to earlier releases. The setting lives in `mu2efastsim.precision` (core). x64 stays enabled in both
modes, so integers, keys, indices and sentinels stay 64-bit; only floating arrays change. Host NumPy code is float64
in both modes.

## Scope

- Precision-aware: the device modules of this package (they call `dprec.register(__name__)` and jit through
  `dprec.jit`): transport, range-out, crossings, tier 1b, photons, SiPM, fused response, waveforms, pulse fit,
  coincidences and the reco chain.
- Not affected: the NumPy chain, and JAX code in other stages (beam, tracker). Shared core kernels (`eloss_jax`)
  compute in float32 only inside these kernels.
- At the hand-off to host code, device results are widened to float64 (`dprec.host`).

## Validation

5 M generated cosmics (100 held-out `gen_even` shards), default configuration (strict_v4, range-out device
transport, offspill, extracted_v04), float32 x2 seeds and float64 x3 seeds, on a V100 and on an RTX PRO 6000 MIG
1g.24gb slice:

- Every validation histogram (seq-0 pulse PE, per-bar rate, earliest-pulse and cluster time to truth, cluster PE,
  layers per cluster) is at the float64 seed floor: the chi2/ndf of float32 vs float64 equals that of float64 vs
  float64.
- Coincidence efficiency (>= 3 layers), float32 minus float64: EX -0.00004 (binomial sigma 0.00011), T1 +0.0006
  (0.0015), T2 +0.0002 (0.0012), M +0.0015 (0.0017). Layer efficiencies agree within +-0.0024.
- Mean seq-0 pulse PE (5-150 PE) is lower in float32: EX -0.032 +- 0.012 %, T1 -0.14 +- 0.06 %, T2
  -0.123 +- 0.016 %, M -0.07 +- 0.07 % (errors from the seed scatter). That is 10-30x smaller than the
  fast-vs-Geant4 differences. Source: see below.

## Reproducibility

**Run to run** (same code, seed and GPU kind): the device chain repeats bit for bit at both precisions. The core
package switches off XLA's GPU autotuning when it is imported (`MU2EFS_DEVICE_DETERMINISTIC`, default on, appends
`--xla_gpu_autotune_level=0` to `XLA_FLAGS`; `MU2EFS_DEVICE_DETERMINISTIC=0` restores autotuning). The setting and the
flag value are recorded in `precision.deterministic`.

Why it is needed: when XLA compiles a fused GPU kernel it times candidate configurations (its native emitter or a
Triton block-level kernel, tile shapes, GEMM configurations) and keeps the fastest. Timings vary between processes,
so the choices do, and with them the order of floating-point sums. In-process repeats always matched; separate
processes did not. Measured 2026-11-01 with autotuning on, held-out shard 22 (50 k generated), seed 1:

| Card | Autotune choices that differ between two runs | What differed between runs |
|---|---|---|
| RTX PRO 6000 MIG 1g.24gb | 52 of 1,408 | float32: transport floats (energy deposit, momentum, path) in some pairs; digi voltages; 1 ADC count of 728 k; cluster PE, time and slope in about 3 % of clusters (up to 0.005 PE, 1.2 ps), same clusters and members. float64: digi voltages only (151 of 722 k samples, 1 ulp); ADCs, pulses and clusters identical |
| A100-SXM4-80GB | 18-26 of about 1,400 | float32: transport floats and digi voltages in 2 of 3 pairs |
| RTX PRO 6000 (full card) | 12-15 of about 1,400 | float32: digi voltages in 2 of 3 pairs |

A floating-point difference upstream can move a discrete result: an earlier float32 pair on the MIG slice changed
the SiPM charges of one shard in ten. With autotuning off, every pair of processes was identical at every stage:
float32 on the MIG slice (3 processes, shards 22 and 24), A100 (2), RTX PRO 6000 (2); float64 on the MIG slice (2).
A shared persistent compilation cache (`JAX_COMPILATION_CACHE_DIR`, used by the UVA kit) also freezes the choices for
the jobs that load from it, but a new cache tunes again. With the setting on, cached and uncached runs were identical.
Stage-by-stage checks: `scripts/chain_identity.py` (per-column hashes of every stage, in-process repeats, dumps and a
per-column diff).

Cost (MIG slice, `e2e-bench` on shard 22 with a reused context, warm-pass chain time without the benchmark's audit,
two interleaved rounds): float32 3.12 -> 3.14 s (+0.8 %, within the 1 % round-to-round spread), float64 9.03 -> 9.07 s
(+0.4 %). The cold pass gets faster because nothing is tuned: 128 -> 108 s (float32), 183 -> 131 s (float64).
`--xla_gpu_deterministic_ops=true` also repeats, but it slows the SiPM stage 3.3x (chain 1.7x): it makes the integer
scatters there deterministic, which they already are. It is not needed because the device code has no floating-point
scatter-add.

**Across GPU kinds** results are not bit-identical. A MIG slice is its own kind: XLA's default configurations depend on
the size of the device, so a slice and the full card of the same chip differ in float values. float32, autotuning off,
shard 22: A100, RTX PRO 6000 and its MIG slice gave the same charges, ADCs and cluster lists, while transport floats,
voltages and cluster PE/time differed. Across architecture classes, float32 discrete counts differ slightly, within the
seed-to-seed spread. For one 50 k-generated shard: 2,866 clusters / 70,459 pulses on Ampere-class cards, 2,869 / 70,593
on a B200, 2,877 / 70,910 on V100 / RTX 2080 Ti. float64 keeps the discrete output across kinds (A100 vs MIG slice,
shard 22: same ADCs, pulses, clusters and members; a ten-card sweep found the same pulse and cluster lists on every
card), but float values differ in the last bits (transport, photon times, SiPM charges, cluster PE and time; pulse PE up
to 3e-7 in the sweep), so it is not bit-identical either. `runchain` (stats, `--emit` shards and merge) and `e2e-bench`
record `precision`: device float, determinism setting, JAX backend and device kinds.

## Speed

float32 makes the six device stages 2-3x faster on cards with a reduced FP64 rate: MIG slice 2.8x, RTX PRO 6000
2.1x, A40 1.5x. It gives nothing on A100, B200 or PVC. Production steady state with a warm JAX compilation cache on
the MIG slice: 698 events with CRV steps per second at float32 vs 261 at float64.

## Source of the pulse-PE offset (bounded, not removed)

Fixed-input stage replay (4 shards, the default configuration on the MIG slice). The float64 input of a stage is fed
to the float32 chain from that stage on. Every `jax.random` sampler computes in float64 from the same key and is
rounded to float32, so the random streams are equal and only the kernels' arithmetic differs. The float64 controls
reproduce the float64 run bit for bit. Mean seq-0 pulse PE (5-150), float32 minus float64:

| Stages in float32 | EX | T1 | T2 | M |
|---|---:|---:|---:|---:|
| pulse fit + coincidences | +0.034 % | +0.000 % | +0.014 % | +0.001 % |
| waveforms, pulse fit, coincidences | +0.036 % | -0.007 % | +0.012 % | -0.000 % |
| photons, SiPM, waveforms, pulse fit, coincidences | +0.044 % | -0.066 % | -0.008 % | +0.018 % |

- On fixed steps, the response and reco kernels do not produce the negative offset: reco and waveforms raise the PE
  by 0.01-0.04 %, and photons + SiPM add -0.07 to +0.01 % with mixed sign.
- `jax.random.binomial` (photon counts) is unbiased in float32 within 0.05 % for n = 5 to 1e5.
- With the same random streams, float32 transport and tier 1b still change the steps of about 95 % of events, at the
  level of individual branch decisions. A per-event comparison is therefore impossible, and the remaining offset
  (about -0.1 %) is attributed to them only statistically.
- Bound: on 5 M generated cosmics, the offset is at most 0.3 % (two sigma) in every sector group. That is 10-30x
  below the fast-vs-Geant4 differences of the default chain. Use `MU2EFS_DEVICE_FLOAT=float64` where a sub-percent
  absolute PE scale matters.
