# CPU-only reconstruction optimization

These controls accelerate the existing NumPy reference. They do not require JAX, a GPU, a compiled extension or a new dependency. Both are opt-in; unset controls retain the original implementation.

```bash
export CRVFS_CPU_ATTACHMENT=numpy
export CRVFS_CPU_FIT_WORKSPACE=1
```

The full CPU chain with these controls was compared on all detailed outputs against a saved reference. A profiled reference is suitable for identity only, **not** for an instrument-free wall-time speedup.

## Exact attachment

`CRVFS_CPU_ATTACHMENT` accepts `reference` (default) or `numpy`. The NumPy path replaces per-event pandas joins with bounded NumPy pair blocks. It uses the native-cluster snapshot, excludes any already-clustered pulse and occupied target side, requires first-pulse anchors and candidates on the same physical counter's opposite end, and retains the strict time window. Candidate choice is nearest cluster with cluster-order ties; losing candidates are not reassigned. One pulse per `(cluster, counter, SiPM)` is selected by distance and original candidate order, then appended in stock key order. No floating reduction or property arithmetic changes.

The reference scalar coincidence combinations remain in use. Broadcasting combinations was not selected: overhead on small groups offset its benefit on the profiled workload. The unchanged exceptional large-event vectorized fallback remains available through its existing interface.

Five alternating same-host CPU replays of retained generated pulses (74,796 pulses, 3,101 clusters) gave median finder time 9.0053 s reference versus 3.0945 s NumPy attachment (2.9101x). All complete ordered output rows, including properties and membership order, were byte-identical in every repeat. This is a finder-only observation; setup, retained-input reads and fingerprint construction were outside timing. It is not an end-to-end or stock-art speedup. Different-host follow-up results must not be combined into the same ratio.

## Pulse-fit workspace

`CRVFS_CPU_FIT_WORKSPACE` accepts `0` (default) or `1`. Two Jacobian buffers are reused across fixed LM and Gauss–Newton iterations. Component multiplications retain their original order; accepted rows are copied into the retained Jacobian. Iteration count, damping, linear solver, Hessian test, threshold decisions, fit recovery and pulse ordering are unchanged.

On retained stock digis (53,523 waveforms, 74,036 pulses), five alternating CPU passes gave about 6.85 s reference versus 6.47 s workspace. Every returned pulse/diagnostic array matched dtype, shape and bytes, including fit parameters. This modest improvement does not establish a general throughput guarantee.

## Current limitation: stock remains faster

The saved Offline v13_41_01 stock job processes the same 2,475-event digi sample with pulse-module total about 1.46 s and finder total about 0.15 s, from TimeTracker means. The CPU NumPy implementation does **not** beat those modules. Stock and candidate observations used different UVA CPU hosts and allocations; these are retained same-input comparisons, not interleaved same-host stock measurements. Framework startup and fast input/setup work are excluded from module/kernel times.

Larger speculative changes (tiling, stalled-state residual scheduling, active-row fitting) were measured but not retained: they added complexity without closing the stock gap. No iteration reductions, changed convergence tolerances, approximate eigensolver, result quantization or relaxed identity gate are used.

## Reproduction and verification

Run under a one-core allocation with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`:

```bash
python3 scripts/bench_cpu_coinc.py --inputs RETAINED_DETAIL.npz --out NEW_FINDER_REPORT.json
python3 scripts/bench_cpu_reco.py --input STOCK_DUMP.root --stock-log STOCK_ART.log --out NEW_PULSE_REPORT.json
```

The finder driver accepts either benchmark detail NPZ or stock `CrvRecoTableDumper` ROOT files. Every timing arm is followed by exact full-row/array checks. The stock-dump pulse driver reuses the existing waveform preparation reader and rejects a stock log with a different event count. Reports are created exclusively, never overwrite evidence. Full-chain candidate validation remains required before promoting either control; isolated speedups are not sufficient.
