# Staged full-chain benchmark workflow

This experimental branch starts at saved CRY primary tables, not live CRY generation. It retains the
existing transport, tier-1b, optical response, waveform, pulse and coincidence physics. The goal is to
measure and reduce implementation overhead without dropping difficult events or changing detector cuts.

## Entry points

- `python3 -m crvfastsim e2e-bench`: provenance-checked repeated full chain; details in `e2e-benchmark.md`.
- `CRVFASTSIM_COMPACT_DISPATCH=1`: optional compact coincidence row batches and sliced host transfers;
  details in `reco-dispatch.md`. Default remains the original dispatch.
- `scripts/bench_rates.py REPORT.json ...`: rates from measured completed pass elapsed time.
- `scripts/compare_bench.py BASE CAND --out NEW.json`: exact same-backend output regression.
  `--allow-derived-rounding` separately accepts only documented tiny derived-cluster changes; it never
  makes a stock-physics certificate.

Inputs, private conditions and output arrays are runtime assets and must not be committed. Each benchmark
refuses an existing output directory. A failed gate can leave diagnostic arrays but never a completed
benchmark report. Keep source immutable until every active invocation has completed its final hash check.

## Run order

1. Pin source, input order, geometry, conditions, library and seed; verify the requested device really exists.
2. Run a small full-chain smoke and a same-process repeat; retain quiet generated events in the denominator.
3. Compare baseline and candidate on identical inputs. Require exact pulse records and membership links;
   evaluate the declared derived-property tolerance explicitly, never merely compare total counts.
4. Run stock-pulse V-R3 with opposite-side attachment enabled, then larger full-chain held-out samples.
5. Apply V-R1/V-R2/V-R4 for changes to response, waveform or pulse arithmetic. A dispatcher-only test does
   not establish whole-chain physical agreement or repair known transport/library modeling residuals.
6. Repeat on each hardware backend. Record GPU model or MIG partition, CPU allocation and whether reco
   uses the CPU. Do not infer an unmeasured backend's speed from device specifications.

## Timing interpretation

The default instrumented pass reloads its context, reads inputs, produces detailed pulses/coincidences,
validates links, hashes assets and writes NPZ. Its cold label means first runchain call in that process;
OS caches are not flushed. Default repeated passes still reload assets; the explicit reuse mode below does not.

Report generated events/s and events-with-steps/s separately. Neither is automatically a stock filter or
trigger acceptance. Do not sum overlapping stage counters, turn cold/warm differences into compiler-only
cost, or subtract setup from full-pass time to advertise a fabricated steady-state throughput. Measure
prepared-context execution directly rather than deriving its rate from default-mode timers.

`e2e-bench --reuse-context` now provides opt-in prepared context reuse with read-only host buffers,
pre/post content audits, fixed device-table identity and separately measured preparation. The existing loop,
input-order RNGs and physics are unchanged. Saved-baseline comparison uses
`scripts/compare_context.py` and requires exact output fingerprints, not a tolerance.
The audited mode includes integrity costs in pass rates; it is not an instrumentation-free production rate.
The comparator remains strict by default. Legacy macOS archives can contain hashed `._*` AppleDouble metadata
sidecars that newer archives omit. The explicit `--allow-removed-appledouble` packaging exception admits only
removed sidecar names with retained paired sources, records every excluded name/hash, and still checks all
paired source hashes and all output fingerprints exactly. It does not allow added/modified sidecars or any
non-setup source changes. Original reports are never rewritten to conceal the packaging difference.

`scripts/context_benchmark.py` extends this check to two input files: shards 12 and 13 with a new seed,
prefetch enabled and compact dispatch. It runs a new reload reference twice, then four prepared passes, and
requires exact fingerprints. Shard 11 runs only prepared passes; existing reload reports can be supplied through
`--reference-dir`, otherwise their comparison is explicitly deferred. `--max-events 50000` applies per input file;
actual full-key denominators are read from completed reports. `--data`, `--lookup-dir`, `--inputs-dir` and explicit
asset paths keep site storage separate from physics. The runner refuses existing outputs and any source drift;
summary directories contain only JSON/logs, while detailed arrays stay under the private `--outdir`.

Next priorities: complete held-out/backend reuse checks; measure waveform lengths and recovery-reason overlaps;
then bounded full-length pulse retries, device packing and pulse hand-offs, response/waveform residency, and
finally generation/transport. Recovery is the union of truncated input, exhausted rounds and fit-window overflow;
its total alone does not measure long-waveform count. Moving recovered rows from NumPy to JAX needs exact
regression plus candidate-path V-R2, including the cohort still recovered. Each change is an independent opt-in
with held-out regression checks. Keep a working baseline throughout.

## Site drivers

Site job scripts for these benchmarks are not kept in the repository. A site run supplies the interpreter or
image, the code snapshot and the private inputs through its own configuration (see `site.env.example`), streams
inputs to node-local temporary storage with SHA256 verification where needed, and records the device, CPU
allocation and whether reconstruction ran on the CPU. Node staging and queue time are not part of the
instrumented pass rate and must be reported separately.
