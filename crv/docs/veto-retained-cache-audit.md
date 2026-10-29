# Retained indexed-cache admission and coverage audit

`veto.retained_cache_audit` exercises the frozen indexed replay loader on the actual
retained inputs without drawing photons or executing detector response. It is a
separate diagnostic, not a production consumer or full physics-campaign admission.
The original registry, batch plan, optical smoke, saved downstream response and
fixed-stock reference are not modified or regenerated.

## Admission and execution

Supply the saved plan directory and its externally recorded summary SHA256. The
runner repeats `retained_plan_report` verification, binds the complete frozen
registry and original timing-census step ledgers, and checks all inherited input,
conditions, code and artifact hashes. It explicitly requires the approved frozen
indexed-loader hash. Mandatory executed modules are copied into the output and
hashed; input and source hashes are checked again before completion. Before the
frozen plan verifier materializes saved Parquets, their metadata is preflighted:
128 MiB file bytes, 2 million rows, 64 columns, 4096 row groups, 256 MiB per
uncompressed row group and 512 MiB summed row-group bytes. Plan table counts and
the current input's step-ledger count are also checked before materialization.
These metadata caps do not bound all decoder allocations or peak RSS. The frozen
plan verifier still performs its existing hash reads; cache load measurements
therefore follow admission and may benefit from warmed filesystem caches. New
binding aggregation rejects conflicts without repeatedly rehashing accumulated
paths. The registry digest is validated before, not inside, measured selections.

```bash
python3 -m crvfastsim.veto.retained_cache_audit \
  --plan /data/retained_plan \
  --expected-plan-sha256 SAVED_SUMMARY_SHA256 \
  --out /data/NEW_CACHE_AUDIT
```

One input cache is opened at a time using the unchanged default loader limits
([contract](veto-indexed-replay.md)). Every proposed batch is selected as a group
of whole frames, preserving the full registry's original labels. This is a loop
over bounded execution batches, not scalar per-event processing. Batch operations
and joins are vectorized. No source-range reads occur during selection. Each
selection is released before the next; cached arrays are closed before opening
another input. Small frame/batch ledgers are retained for final persistence.

The audit checks:

- Every all-event, step and particle source row is visited **exactly once**. Three
  byte-per-row bitmaps reject duplicate visits and incomplete final coverage.
- Exact full input-SHA/run/subrun/event/frame identities, requested frame order,
  saved phase/intensity, and original all-event row numbers.
- Every raw selected column against its original row in the immutable typed
  cache. This deliberately reads the pinned loader's private array interface; a
  loader API change requires a new reviewed audit version.
- Original source ordering for steps and particles, and an independent comparison
  of IDs/bar/endpoints/energy against the previously saved timing census.
- Full-tuple step-to-particle joins and exact joined PDGs, including unused
  particles and empty frames rather than only nonempty response support.
- Per-frame raw/finite endpoint counts against the frozen plan and exact batch
  counts and typed-selection byte accounting.

Per-batch typed-value SHA256 fingerprints include original source rows and frame
identity. NaNs and signed zero are normalized for these fingerprints only; actual
field comparisons use exact numerical equality with NaNs equal. These hashes are
not a bitwise NaN-payload claim or independently regenerated physics. Particle
source identities describe synthetic replay particles, not independent upstream
showers or physical ancestry.

## Output and memory measurements

One exclusive `BatchArtifacts` transaction saves the full registry, frame ledger,
batch ledger, admission receipt, source snapshots and summary. It has an **8 MiB
logical pathname-byte cap**, including status/receipts and failure reserve. No raw
cache, selected step tables, ROOT file copy or response products are persisted.
Failures leave incomplete evidence; no resume, truncation, dropped frame,
denominator change or overwrite is permitted. The atomic-publication and
final-name crash-durability limitations of [the artifact protocol](veto-batch-artifacts.md)
apply unchanged.

The summary records input load time, selection-plus-validation time, total time,
exact resident NumPy payload and coverage-bitmap bytes. Linux current RSS is
sampled before loading, after loading, after selection auditing, and after cache
closure/garbage collection. `ru_maxrss` is normalized to bytes and explicitly
reported as a **cumulative process high-water mark**. It includes imports,
saved-plan verification, preceding input allocations and allocator retention; it
is not an isolated per-input peak or the sum of resident-array sizes. Use external
`/usr/bin/time -v` for the entire process including final writes/commit. Neither
this measurement nor any payload/output limit is a process-memory hard cap.

The saved-only `veto.retained_cache_report` verifies protocol, hashes, required
source bindings and original plan admission, and reconstructs frame/batch census
and byte formulas without reopening a cache or regenerating response. Source-row
bitmaps, raw-table fingerprints and measured memory/timings remain hash-bound
execution measurements: the saved verifier cannot independently reproduce raw
joins from summary fields alone.

The transaction's seed zero is a required protocol placeholder for a read-only
operation; it drives no RNG and adds no exposure. `execution_admitted=false`,
`optics_executed=false`, `response_executed=false`, `F=null` and
`finite_library_uncertainty=null` remain explicit. This audit can close the tested
input-cache and planned-selection coverage gate only. Streaming keyed optics plus
bounded downstream persistence, coincidence/native membership, frozen-track
coverage, output admission and exposure/uncertainty gates remain separate.
