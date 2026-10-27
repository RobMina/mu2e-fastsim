# Opt-in response execution observation

`crvfastsim.response.execution_trace` observes the existing response functions
without replacing photon or SiPM implementations. It is a synchronous diagnostic,
not a receipt, backend selection mechanism, or execution authentication scheme.
It does not change RNG state, response configuration, or JAX placement policy.
Nothing is installed merely by importing the module.

## API

```python
from crvfastsim.response import execution_trace

with execution_trace.observe("jax") as trace:
    D = fast_digis(...)  # Configure the requested backend separately.
evidence = trace.summary()
evidence = trace.require_backend()  # ValueError unless the requested path is confirmed.
```

Accepted names are `numpy`, `jax` (nonfused), and `fused`. Contexts are single-use.
Call `require_backend()` after leaving the context and before admitting diagnostic
outputs. A strict caller must not silently catch its `ValueError`. `summary()` is
JSON-safe, returns fresh containers, and can also describe incomplete/failed
observations. Neither method depends on the profiler still being installed.

The observer uses `sys.setprofile` Python `call` events, matching the actual loaded
response function's code object as well as its exact module and name. Counters
include:

| Requested path | Required observed functions |
| --- | --- |
| NumPy | `photons.make_photons`, `photons._substep_photons`, `sipm.simulate`, `sipm._simulate_chunk` |
| Nonfused JAX | `photons_jax.make_photons`, `photons_jax.substep_photons`, `sipm_jax.simulate`, `sipm_jax.run_device_sipm` |
| Fused JAX | `device.response_batched`, `device.response`, `photons_jax.device_photons`, `sipm_jax.run_device_sipm` |

These are Python entry points, not hardware instruction or JIT-executable counts.
Warm JIT caches do not remove the required Python dispatcher calls. Shared NumPy
row preparation in JAX response is expected and is not classified as fallback.

Required calls alone are not sufficient. Confirmation also requires a normally
completed context, no unsupported use, no incompatible response path, nonempty
photon rows, and an actual nonempty SiPM workload. NumPy SiPM workload is observed
at `_simulate_chunk`; JAX SiPM workload is measured from the existing immediate
caller's `n` (nonfused) or `nv` (fused) when `run_device_sipm` is called. These
implementation-specific observations deliberately fail closed if the dispatch
structure changes. A bare standalone device-SiPM call is not a complete response
path.

In particular, fused execution may dispatch a padded SiPM buffer when every
sensor is absent. That does **not** confirm useful SiPM workload or absence of
fallback. Empty input, no photons, all-status-masked input, and missing kernel
paths remain insufficient evidence. Zero *output charges* alone does not mean
no workload: genuine photon/SiPM dispatch can legitimately produce no charges.

Any `sipm.simulate` call while requesting `jax` or `fused` records NumPy SiPM
fallback and makes `require_backend()` fail, even if the JAX dispatchers also ran.
This covers the existing nonfused wrapper's overflow fallback conservatively;
it does not infer why NumPy was called.

## Evidence fields

- `schema_version: 1`, `observer: "python_sys_setprofile"`, `requested_backend`.
- `status`: `confirmed`, `fallback`, `insufficient_evidence`, or `unsupported`.
- `backend_confirmed`: the strict decision; `require_backend()` returns the same
  evidence only when this is true.
- `calls`: per-symbol Python call counts, using the names in the table above.
- `required_calls`, `missing_calls`, `unexpected_calls`, `reasons`, `unsupported`.
- `photon_workload_calls`, `sipm_workload_calls`, and `completed`.
- `fallback_detected`: whether a NumPy SiPM entry was observed for a JAX request.
  False alone is **not** evidence excluding fallback. `fallback_observation` is
  `observed`, `not_observed_in_observed_calls`, `insufficient_evidence`, or
  `not_applicable` (NumPy). The qualified non-observation result is issued only
  for a confirmed complete path.
- `limitations`: the interpretation limits carried with the record.

Only JAX/fused requests import JAX and add a `jax` block: `version`,
`available_devices`, `observed_argument_devices`, and `placement_evidence`.
Device records contain the actual `platform`, `id`, `device_kind`, and description.
The observed list comes from JAX array arguments at the instrumented entry points,
not from an assumption that the first available device ran the work. Available
accelerators are not execution evidence. CPU placement is recorded as `cpu`, never
GPU. Argument placement does not prove every response operation used that device.
NumPy observation neither imports JAX nor emits JAX metadata.

## Supported scope and profiler ownership

This implementation supports CPython synchronous execution in the main Python
thread only. It rejects an existing current-thread profiler, an existing
`threading` profiler, nesting, worker-thread entry, or active Python workers
(including raw `_thread` workers) before installation. Attempts to start Python
threads while observing are rejected and permanently invalidate the trace, even
if the caller catches the exception. Native JAX/BLAS runtime workers are outside
this Python-thread observation boundary and are not themselves rejected.

The observer removes its profiler on normal completion and exceptions. If another
component replaces the hook during observation, the trace is invalidated and the
replacement is left intact rather than overwritten. Existing profilers are never
temporarily suspended. Changing profile hooks, exiting from another thread,
background callbacks, multiprocessing, and asynchronous response scheduling are
unsupported. Use the observer around the synchronous `fast_digis` work, not
around worker/prefetch scheduling.

## Limits

This is opt-in in-process call evidence. It does not authenticate code, bind input
or output assets, prove every operation's architecture, trace compiled/native
kernels, establish physics closure, or certify GPU execution. An instrumented
process can tamper with its own evidence. JAX may dispatch asynchronously; the
observer does not add synchronization or a device-level profiler. Its all-Python
call-hook overhead means timings collected under observation are diagnostic,
not benchmark timings. Traced/fused diagnostic fasttables runs remain outside
normal receipt admission initially; this module itself does not create receipts.

## Fasttables integration

Named step-mode `fasttables --trace-response` brackets the response/waveform
calculation and calls `require_backend()` before writing ROOT or metadata.
It records the returned `response_execution` evidence. `--fused-response` is a
diagnostic requiring `--jax --trace-response` and the fixed offspill window.
Reco-only, legacy mode, receipt admission and reuse of output paths are refused.
Untraced defaults are unchanged. All three response modes still use the same
NumPy pulse/coincidence reconstruction in fasttables.

The trace requires local ROOT inputs. During a traced invocation only, the CLI
scopes Uproot's documented handler defaults to `MemmapSource` with threading off,
restoring the defaults on normal or exceptional exit. Default fsspec local reads
otherwise leave unrelated `fsspecIO`/asyncio workers alive after file closure,
which the observer correctly refuses. The CLI does not whitelist those workers,
terminate other threads or weaken the response-thread rule. Network ROOT URLs
are outside this mode. The metadata records this local-I/O policy. Native JAX
workers remain outside the Python-call observation boundary.

`bash scripts/run_named_response_seed.sh NEW BASE STEPS CORE run2 jax SEED`
provides a serial driver. Set `CRVFS_LOOKUP_DIR` and
`CRVFS_VALIDATION_OFFLINE` to explicit existing optical and Offline payload
roots. It records input/optical/source hashes before execution and rechecks them
afterward, writes a paired saved-product audit, and marks completion only after
success. `analyses.named_response_assets` resolves the actual counters/waveform
through the producer's resolver and requires them to equal the explicit Offline
assets; a working-directory `Offline/...` shadow is rejected even if bytes match.
It also inventories every configured lookup and photon map. The driver compares
these resolved path/hash/size inventories before and after production; this is
not an atomic snapshot or execution attestation. Linux execution is pinned to one allowed CPU; macOS uses host CPU with
no affinity claim. The driver does not install software, fetch data, build,
submit jobs or provide a normal response receipt. Run one replay at a time.

## Focused tests

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu \
PYTHONPATH=/tmp/r1-core-233771b:$PWD \
/Users/rmina/mu2e/.venv/bin/python3 -m pytest -p no:cacheprovider -q -s \
  tests/test_response_execution_trace.py
```

Run this file in its own pytest process. Policy tests cover profiler restoration,
exceptions, nesting, foreign-profiler preservation, threading refusal, empty or
wrapper-only evidence, exact function matching, actual NumPy SiPM fallback
observation, and no JAX imports for NumPy mode. Tiny in-memory optical tables are
reused from `test_readout_execution.py`; real `chain.charges` runs exercise all
three untouched response backends, compare traced/untraced outputs and RNG state,
and check all-absent sensors. No response function or loader is patched. JAX tests
explicitly select CPU and skip only if JAX is unavailable. These synthetic tests
are not production optical-payload, GPU, or independent-seed physics validation.

The focused local run on 2026-10-27 passed **26 tests in 1.93 s**, with no skips,
including nine real charge executions (untraced, traced, and all-absent for each
backend). JAX 0.11.2 reported observed argument placement on `cpu:0`. The complete
local test log is `/tmp/response-execution-trace-tests.log`; this is a test result,
not a performance benchmark or production-backend certificate.
