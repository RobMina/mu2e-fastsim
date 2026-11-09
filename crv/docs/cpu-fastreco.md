# CPU reconstruction speed

Two exact speedups of the NumPy reference are on by default. They need no JAX, GPU, compiled extension or new
dependency, and their output is byte-identical to the original code. The variables below only exist to measure them
and to fall back: `CRVFS_CPU_ATTACHMENT=reference` and `CRVFS_CPU_FIT_NORMAL_CACHE=0` select the original code.

Interleaved passes of the NumPy chain (shards 22 and 24, 100,000 generated events, one A100-80GB node's CPU, four rounds;
ratios are paired, ranges are over rounds): attachment 1.154x (1.148-1.161), normal-equation cache 1.028x (1.026-1.035),
both 1.19x, with reco time 26.7 -> 14.1 s of a 79.9 s pass. The outputs were byte-identical on ten held-out shards at
two seeds. On the GPU chain only the NumPy redo of flagged pulses uses them: 1.015x (1.013-1.018). See the switch table
in [validation.md](validation.md#speed-switches).

## Exact attachment

`CRVFS_CPU_ATTACHMENT` accepts `numpy` (default) or `reference`. The NumPy path replaces per-event pandas joins with bounded NumPy pair blocks. It uses the native-cluster snapshot, excludes any already-clustered pulse and occupied target side, requires first-pulse anchors and candidates on the same physical counter's opposite end, and retains the strict time window. Candidate choice is nearest cluster with cluster-order ties; losing candidates are not reassigned. One pulse per `(cluster, counter, SiPM)` is selected by distance and original candidate order, then appended in stock key order. No floating reduction or property arithmetic changes.

The reference scalar coincidence combinations remain in use. Broadcasting combinations was not selected: overhead on small groups offset its benefit on the profiled workload. The unchanged exceptional large-event vectorized fallback remains available through its existing interface.

Five alternating same-host CPU replays of retained generated pulses (74,796 pulses, 3,101 clusters) gave median finder time 9.0053 s reference versus 3.0945 s NumPy attachment (2.9101x). All complete ordered output rows, including properties and membership order, were byte-identical in every repeat. This is a finder-only observation; setup, retained-input reads and fingerprint construction were outside timing. It is not an end-to-end or stock-art speedup. Different-host follow-up results must not be combined into the same ratio.

## Normal-equation cache

`CRVFS_CPU_FIT_NORMAL_CACHE` accepts `1` (default) or `0`. The fit recomputes JTJ and JTr only for rows whose residual
and Jacobian changed in the last iteration. Iteration count, damping, linear solver, Hessian test, threshold decisions,
fit recovery and pulse ordering are unchanged.

## Current limitation: stock remains faster

The saved Offline v13_41_01 stock job processes the same 2,475-event digi sample with pulse-module total about 1.46 s and finder total about 0.15 s, from TimeTracker means. The CPU NumPy implementation does **not** beat those modules. Stock and candidate observations used different UVA CPU hosts and allocations; these are retained same-input comparisons, not interleaved same-host stock measurements. Framework startup and fast input/setup work are excluded from module/kernel times.

Larger speculative changes (tiling, stalled-state residual scheduling, active-row fitting) were measured but not retained: they added complexity without closing the stock gap. No iteration reductions, changed convergence tolerances, approximate eigensolver, result quantization or relaxed identity gate are used.

## Reproduction and verification

Run under a one-core allocation with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`:

```bash
python3 scripts/bench_cpu_coinc.py --inputs RETAINED_DETAIL.npz --out NEW_FINDER_REPORT.json
python3 scripts/bench_cpu_reco.py --input STOCK_DUMP.root --stock-log STOCK_ART.log --out NEW_PULSE_REPORT.json
```

The finder driver accepts either benchmark detail NPZ or stock `CrvRecoTableDumper` ROOT files. Every timing arm is followed by exact full-row/array checks. The stock-dump pulse driver reuses the existing waveform preparation reader and rejects a stock log with a different event count. Reports are created exclusively, never overwrite evidence. `scripts/switch_ab.py` is the full-chain A/B: it alternates the arms inside one prepared process and fails on any output difference.
