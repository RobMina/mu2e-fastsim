# Parallel prepared-context content auditing

Prepared chain reuse keeps geometry, conditions, the tier-1b library, optical
lookup tables and cached templates between sequential calls. Its pre/post-pass
content audit detects accidental nested configuration changes, pandas column
replacement, mutable aliases and buffer edits. File provenance remains checked
at preparation and at final benchmark verification. A context is single-owner;
these checks are not a security boundary or support for concurrent mutation.

## Opt-in execution setting

`CRVFS_CONTEXT_HASH_WORKERS=N` enables up to N hashing threads, with 1 the
unchanged serial default and a supported range of 1–32. It changes neither the
hash schema nor any simulation/reconstruction operation. For example:

```bash
CRVFS_CONTEXT_HASH_WORKERS=4 python3 -m crvfastsim e2e-bench ... --reuse-context
```

The setting is captured by the benchmark's existing CRVFS environment inventory.
Use it only within the CPU allocation: four threads per process across eight
GPUs requires capacity for 32 hash workers, in addition to other active host
work. Keep 1 on a single-core interactive host. More threads are not necessarily
faster under whole-node memory-bandwidth contention.

## Identical guarantee

Traversal, type validation and pandas extraction stay on the calling thread.
Numeric NumPy leaves of at least 1 MiB are dispatched to bounded worker threads;
small leaves, strings and metadata keep their existing treatment. Each SHA256
node still receives the same ordered length-prefixed bytes and child digests.
Parents resolve deferred leaf digests in traversal order, yielding the identical
serial fingerprint. Shared references stay memoized only within that one call.
No digest is reused across audits, no bytes are skipped and no read-only or
allocation-identity shortcut substitutes for a host content hash. Writable
aliases remain detectable. JAX arrays retain their existing synchronization,
immutable identity/placement check and no-device-to-host-copy contract.

The optimization leaves asset loading, pinned hashes, sidecar validation,
configuration signatures, global-cache identity checks and device admission
untouched. There is no checkpoint, persistent cache or mmap format to invalidate.
A thread pool exists only for the duration of one fingerprint call.

## Measurement tooling

`scripts/profile_prepared_chain.py` invokes the existing prepared benchmark and
optionally records cProfile files for preparation and each pass. Profiled wall
times are diagnostics, not throughput measurements. `scripts/profile_context_hash.py` compares exact
complete-context digests with 1/2/4/8 workers over three repetitions.
`scripts/report_prepared_profile.py` summarizes retained reports without reruns.
The saved-baseline comparator remains `scripts/agent13_compare_context.py`.

One CPU-node complete-payload measurement gave per-audit three-repeat medians
8.7550/4.4521/2.3940/1.3664 seconds at 1/2/4/8 workers, with exact serial digests
and 4.05 GiB peak process RSS. This is a hash-only workload, not a chain speedup
or a measurement on a GPVM, A100, Polaris or Aurora. The optical host buffers
are much larger in a NumPy context than the retained device-array metadata of
a JAX context, so this speedup must not be transferred to GPU-chain timing.
On one A100-80GB, ten fixed cached-primary shards (500,000 generated events,
22,432 events with steps), NumPy transport/tier-1b and GPU response/reco, the
three-repeat warm median changed from 80.2892 s [80.2848, 80.5955] to 77.7179 s
[77.6355, 77.8442] with four workers: an observed 1.0331x chain speedup and
279.390 to 288.634 events with steps/s. Complete pre/post audits changed from
4.9332 to 1.5113 s per pass. All retained named-array outputs are byte-identical
to the saved serial baseline, with unchanged conditions and physics sources.
Preparation was 55.901 versus 54.432 s; library setup itself is not optimized.
The observations are sequential, not interleaved timing trials or a confidence
interval. Cold passes were separately 220.282 and 216.437 s. No audit-subtracted
rate is reported, and this does not certify concurrently active whole-node or
cross-site throughput.

A preparation profile attributes most setup time to library topology attachment
and library reading. Those are not changed by this optimization. A future
prepared-topology cache would require hashes binding the library, sidecars,
geometry, attachment implementation and complete resulting content; bypassing
existing geometry/sidecar checks is not an acceptable cache key.
