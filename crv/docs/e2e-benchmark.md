# Stage-0 reproducible full-chain benchmark

## Boundary and invocation

This is instrumentation of **the existing `analyses.runchain.run`**, not an
alternate physics implementation. The boundary is explicitly:

**cached CRY `dump/gen` tables → stage-C transport → tier-1b library → photons /
SiPM charges → digis → packed reco waveforms → pulses + coincidence clusters**.

CRY generation, Geant4, art startup and stock reconstruction are **not timed**.
No stock acceptance or speedup is inferred. The denominator is the actual unique
`(run, subrun, event)` universe in the input gen tables, including generated
events that produce no steps, pulses or clusters. This relies on the existing
cached-gen interchange contract that every generated event has a gen row;
events absent from the supplied tables cannot be reconstructed or inferred.

```bash
PYTHONPATH=/path/to/core:/path/to/crv python3 -m crvfastsim.analyses.e2e_bench \
  '/private/gen.*.root' --library /private/library.root \
  --calib /private/calib.csv --deviation /private/deviation.csv \
  --gdml /private/world.gdml --outdir /private/new-benchmark \
  --jax --seed 20261027 --repeat 3
```

`--outdir` must be explicit and **must not exist**, even as a dangling symlink.
`--repeat` is the total number of passes: first `cold`, then same-process `warm`
passes. Omit `--jax` for NumPy CPU. All existing runchain physics options are
accepted, including configuration, window, transport switches, counters,
`--max-events`, prefetch, fused response and `--reco-cpu`. `--compare` and
`--emit` are recognized but refused: stock-denominator validation and external
merge products are outside this stage-0 contract. Unknown Python API options
are refused, not silently ignored.

Inputs, source and conditions must already be local. The benchmark disables
geometry's optional asset fetching for its duration. It does not install,
build, submit, copy to remote hosts or change a source checkout. It never writes
input tables. Output products contain internal simulation data and must stay
in private storage; do not commit or publish them.

## APIs

* `e2e_bench.run(inputs, outdir, repeat=2, seed=1, **runchain_options) -> dict`:
  admission, provenance, repeated existing runchain calls, identity validation,
  and exclusive final report creation.
* `e2e_bench.main(argv=None)`: standalone module CLI.
* `runchain.run(..., benchmark=None)`: the only extension of the existing entry
  point. With no callback, no detail tables, hashes, device probes or benchmark
  validation are requested. A callback implements:
  * `setup(rc, cfg, gdml, library, calib, deviation)` before asset loading;
  * `check_input(index, path, gen)` immediately after input read and **before**
    any max-events selection;
  * `output(index, path, selected_keys, outs, *, st, pa, charges, digis,
    waveforms, timers)` after detailed reconstruction for that input.
* `Recorder` implements that callback; it is not a physics API or an admission
  substitute. Use `e2e_bench.run` for complete benchmark reports.
* `event_keys`, `detail_arrays`, `fingerprint` expose the small validation and
  retention primitives used by focused tests.

### Opt-in prepared context

`--reuse-context` retains loaded setup across passes; the default still reloads it. `runchain.prepare(...)`
creates a single-owner `PreparedContext`, and `runchain.run(..., prepared=context)` uses it. Setup arguments
must match, including resolved paths, configuration, world/absorber switches and asset-search environment.
RNGs, input tables, outputs, accumulators and prefetch executors are never stored in the context.

Preparation deep-copies configurations, loads geometry/conditions/world/library/optics, prewarms the optical
propagation and waveform-template caches, marks reachable NumPy buffers read-only and fingerprints their
contents. Audits before and after each pass detect nested/config/DataFrame mutations; retained JAX buffers
are checked by immutable array identity and placement without device-to-host copies. Python structures are
not a security boundary: callers must not modify or share a context concurrently. Creating another context
or clearing the process-global waveform/optical cache invalidates an existing context. This API is for
sequential repeated workloads, not concurrent configurations or an optimizer's mutable parameter object.

`CRVFS_CONTEXT_HASH_WORKERS=4` optionally parallelizes large host-array hashes without changing the serial
fingerprint or skipping content checks; the default is one worker. See [audit worker settings](context-hash-workers.md)
for CPU-allocation limits, measurements and the exact guarantee.

Memory-dependent fused-response batching is explicitly **unsupported** with reuse. Reuse retains the existing
two-stage response, transport and all recovery paths. It does not change kernel shapes or RNG semantics.

The report labels `context_policy=prepared_once`, records `preparation_elapsed_s`, preparation stage/hash
costs and the process-local context fingerprint. `cold` then means first execution **after preparation**, not
first setup-plus-execution. Each measured pass includes pre/post context audits (`context_audit_s`), input IO,
full physics, detailed validation/NPZ and stats. Asset content is hashed at preparation and final verification;
reused passes validate paths/config rather than rehash immutable files. Source/input/output final checks remain.
`passes_elapsed_s` and `invocation_elapsed_before_report_s` are measured outer intervals, not summed timers;
the latter includes preparation and final verification. No audit-subtracted throughput is advertised.

`scripts/compare_context.py OLD_REPORT NEW_REPORT --out NEW.json` reuses a completed baseline. It
requires exact retained fingerprints (no rounding tolerance), matching devices/versions/config/input order,
matching asset contents despite relocated paths and unchanged non-setup package sources. It allows only the
three setup/instrumentation source files to differ and records those changes. Timings against saved reports
are non-interleaved observations, not measurements of timing variance on the old arm.

No event/pulse Python loop was added by instrumentation. Loops cover files,
reconstruction chunks, fields, sectors and repeats. Input-order RNG behavior
remains `np.random.default_rng([seed, input_index])`; no alternate sampling or
reconstruction is introduced. The response's existing JAX key behavior is
preserved, including its default key where runchain does not supply one.

## Retained artifacts

A successful output directory contains:

```text
benchmark.json
pass_000/runchain_stats.json
pass_000/input_0000.npz
pass_000/input_0001.npz
pass_001/runchain_stats.json
pass_001/input_0000.npz
...
```

`benchmark.json` is only created after all passes, fingerprint comparisons and
final input/asset/source/output hash checks succeed. An interrupted or rejected
run may leave diagnostic partial pass directories, but **no completed report**.
There is no resume or overwrite mode. A stats file alone is not a completed
benchmark. Hashes bind integrity, not authenticity, and do not prevent an
adversary rewriting both products and manifest.

Each per-input NPZ uses ordinary non-object NumPy arrays (`allow_pickle=False`):

* `input_generated_keys`: full unique input event universe before selection.
* `generated_keys` and `generated_packed`: complete selected universe used in
  the denominator, not merely events with activity.
* `rng_seed_and_input_index`: actual seed and input-order index, retained in
  every pass. Separate invocations with different seeds each retain their own
  event lists; repeats deliberately **do not** change the seed.
* `events_with_steps_keys`, `events_with_particles_keys`,
  `events_with_charges_keys`, `events_with_digis_keys`,
  `events_with_waveforms_keys`, `events_with_pulses_keys`, and
  `events_with_clusters_keys`: exact full-key activity lists, not acceptance.
* `chunk_count`; `chunk_0000__cl_*`, `chunk_0000__mem_*`,
  `chunk_0000__pulses__*`, and any other raw reconstruction fields. Chunk-local
  cluster references are deliberately not renumbered or rematched.
* `chunk_0000__cluster_keys`, `...__member_keys`, `...__pulse_keys`: full
  `(run, subrun, event)` keys for every retained raw output row.
* `count__*`: all available output counts, also included in fingerprints.

The raw `detail=True` interface supplies cluster position and scalar fields,
membership, and flat pulse `ev,idx,bar,sipm,seq,pe,time`. This is **not** a new
stock EventNtuple exporter: fit parameters and waveform-membership links not
provided by that interface are not fabricated. Existing `reco.ntview` remains
the place for its established views. This module does not duplicate it.

The report records per-input counts of steps, particles, photons when exposed,
charges, digis, reconstructed waveforms, pulses, members, clusters, fallback
events and activity events. All observed sector types are kept, including side
sectors; configured types with no clusters receive zero counts. Sector types
are not claimed to identify individual physical sectors when the underlying
chain groups several sectors into a type. **Fused response does not expose its
photon count:** `photons` is JSON `null` there, not zero or an estimate.
Intermediate charge/digi/waveform tables are counted and validated but not
persisted; the raw retained detailed physics tables are pulses/members/clusters.

## Fail-closed checks

* Empty/missing/duplicate input files and empty gen tables are rejected.
* Keys must be nonnegative integers. Input event values must fit the existing
  **20-bit** chain event field; packed channel/FEB arithmetic must fit int64.
* The chain omits `run` from its packed event labels. Different full keys with
  the same packed `(subrun,event)` are therefore rejected across the entire
  input set, as are duplicate full events in separate input files. Multiple
  primary rows belonging to one full event are expected, not duplicates.
  Different runs with disjoint packed labels are permitted.
* Full inputs are checked before `--max-events`; selection cannot hide a
  collision. Existing selection semantics are retained: first N sorted unique
  **event numbers per file**, which may select more than N full keys when
  event numbers recur in different subruns. The exact full selection is saved.
* Step/particle/charge/digi/waveform/cluster/member/pulse events must belong to
  the selected generated universe. Columns must have consistent lengths.
* Cluster references must be in range and in the member's event; pulse
  `(event,idx)` references must exist; indices must be contiguous and unique;
  duplicate membership edges and duplicate pulse identities across chunks fail.
* Residual truncation/overflow/drop flags or counts fail. Pulse/charge/digi
  table counts must agree with existing stage counters. Successful full-length
  pulse retries and coincidence fallback are permitted and recorded; they
  are not residual truncation. These checks rely on the existing physics
  implementations exposing overflow and performing their documented retries;
  instrumentation cannot detect an unreported upstream loss of input rows.
* `--jax` means an actual accelerator is required: a synchronized probe of
  default JAX array placement must be GPU/oneAPI, not CPU; subsequent calls
  are pinned to that device. Missing plugins or CPU default placement fail.
  `--reco-cpu` remains an **explicitly labelled mixed-device run**, never an
  all-GPU claim. Existing kernel CPU fallbacks also remain in the pipeline.
* Repeats must have identical named-array fingerprints, including dtype,
  shape, bytes, counts, raw tables and generated-event lists. This is stronger
  than aggregate equality, intentionally sensitive to output order. No
  numerical tolerance or event rematching conceals a mismatch. It establishes
  repeat closure on that device, not CPU/GPU or physics equivalence.

## Timing and provenance

The throughput denominator is each pass's **measured elapsed wall time** around
`runchain.run`, including setup, input IO, asset hashing, packing, detailed
output validation/hashing/writing, stats writing and printing. Rates are
`generated_events / elapsed_s`, never sums of overlapping stage times.
Prefetch and pipelined JAX reco can overlap CPU/device work. Raw per-file phase
wall timers are diagnostic, are sometimes nested, and must not be added to
claim a chain rate. GPU output collection occurs before detailed output is
saved; per-kernel asynchronous dispatch timers are not kernel execution times.

New diagnostic timers include `geometry_conditions_config_s`, `input_io_s`,
`waveform_pack_s`, `reco_s`, `detail_output_s` and `stats_output_s`, alongside
existing setup/transport/tier1b/response/reco subtimers. The final benchmark
report contains `stats_output_s`; the earlier runchain stats file cannot include
its own completed-write time. Per-file detail timers include validation,
content hashing, NPZ IO and NPZ hash. Device admission and initial input/source
hashing occur before the timed pass, with their own preflight timers; final
provenance/output verification is reported separately. Final report write and
Python imports are outside the reported pass wall and explicitly not called
physics throughput. No separate cache-free or instrumentation-free rate is
invented.

`cold` means the first **runchain call in this process**. Device discovery/probe
already occurred, and input/source hashing may have warmed filesystem caches.
No OS cache flush, persistent JIT cache reset, or fresh interpreter is performed
between repeats. Without `--reuse-context`, `warm` retains process/JIT/template caches but runs the same
runchain setup and IO again. With reuse, preparation has its own measured interval and the existing
runchain loop consumes the same sealed setup objects. Repeated pass times and identical fingerprints are reported
individually, not pooled into a claimed GPU speedup.

Provenance includes SHA256s of all inputs, tier-1b library, resolved GDML,
counters, conditions/deviation CSVs (unless nominal), all configured photon
LUTs, photon map, and single-PE template. Effective run/chain dataclasses are
recorded; `null` optional subconfigs mean the defaults in the hashed sources.
All `.py` and `.json` files under the imported CRV and core packages are hashed
before measurement and verified afterward, without Git commands. Installed
third-party package versions, Python executable/version, OS/hostname/CPU count,
architecture, JAX device kind/platform/backend version/precision (both before
and after chain imports, which can enable x64) and relevant
backend/thread/asset environment settings are recorded. This is not a container
or binary-library digest, power trace, GPU utilization profile, hardware
occupancy measurement or clock-stability certification.

## Synthetic tests and limits

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=<core checkout>:<crv checkout> \
python3 -m pytest -q -p no:cacheprovider \
  tests/test_e2e_bench.py tests/test_chain.py
```

The focused command passed **53 tests** locally. Including
`test_chain_config_wiring.py`, `test_chain_reco_contract.py` and
`test_response_contract.py` passed **141 tests** in the owned-core environment.

Tests use synthetic input/asset bytes and monkeypatched physics stages while
running the actual runchain loop, callback, retention and orchestration. They
exercise repeat closure, zero-output events, max-events census, raw chunk-local
references, all-sector retention, non-benchmark equivalence, collisions,
truncation, provenance drift, no-clobber and GPU-to-CPU refusal. They establish
instrumentation contracts, **not** a real-data rate, GPU correctness, physical
acceptance or physics agreement. Real full-chain/GPU timing requires authorized
local detector assets and must be reported separately.
