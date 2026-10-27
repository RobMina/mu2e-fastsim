# Optional compact reconstruction dispatch

## Scope and API

This is an experimental **host-dispatch/transfer optimization**, not a physics change or a device-resident
reconstruction pipeline. Enable it before reconstruction with:

```bash
export CRVFASTSIM_COMPACT_DISPATCH=1
```

Unset the variable, or set it to any other value, to retain the baseline fixed-tier dispatch and collection paths.
There are no new public function arguments, response knobs, or changed defaults. The opt-in affects
`reco.coinc_jax.find_async` / `finish` / `find` and the pulse collection within `reco.chain_jax.run`.
`with_pos`, `keep_pulses`, attachment configuration, existing tier overrides, and fallback controls retain their
meanings. Each pending pulse dispatch captures its mode; a coincidence handle captures its mode through
collection and all promotions. Changing the environment while a handle is pending cannot change its transfer or
promotion policy. Set the variable once per run rather than toggling it between chunks.

Only `crvfastsim/reco/coinc_jax.py`, `crvfastsim/reco/chain_jax.py`, `tests/test_reco_dispatch.py`, and this document
are in scope.

## Stage 1 implementation

- A coincidence batch with `n` real group rows and existing tier cap `B` uses row capacity
  `min(B, 2**ceil(log2(n)))`. Full batches remain `B`; only partial batches shrink. An exceptional non-power-of-two
  cap is never rounded upward. For example, three rows in the existing 16,384-row tier use capacity four.
  Capacity choices are bounded by the existing batch cap, not one compilation per exact occupancy.
- Compact collection slices every requested output leaf to `[:n]` **before** host conversion. All requested leaves
  of all batches in that collection are passed to one `jax.device_get` pytree call; host concatenation follows.
  This applies to coincidences and pulses. Cluster positions are included only with `with_pos=True`.
- **Pulse kernel batch shapes remain fixed.** An initial experiment shrinking pulse-fit row capacities produced
  a one-ULP PE difference on CPU (`30.026008209753822` versus `30.026008209753826`). That shape change was removed,
  not hidden behind a tolerance. Pulse dispatch therefore keeps its original padding and only optimizes output
  transfer. Coincidence tier row capacities are the capacity optimization in this stage.

`find_dense` and its physics arithmetic are unchanged. H, K, tier assignment, split criteria, attachment,
root selection, and promotion/fallback decisions are unchanged. Existing oneAPI removal of H=32 tiers and the
four-layer K<=32 cap remain in force; row-capacity selection never exceeds the supplied tier batch cap.

## Index and overflow invariants

Only the **leading group/waveform row dimension** is sliced. Coincidence labels remain hit-column root indices;
no hit slots are removed, relabelled, or flattened. Group/event maps, original pulse indices, and per-root positions
stay aligned. In pulse collection, concatenation retains input row order, all pulse slots and sequence indices;
`rows[wf]` still maps through the original waveform selection, including nonidentity row selections.

Every real-row `overflow`, `rounds_exhausted`, `window_overflow`, and `input_truncated` flag survives collection.
Host-side `input_truncated` is included in the same output tree. Padding flags cannot request recovery for
nonexistent rows. Existing higher-H/K promotions and full-event NumPy coincidence replacement remain unchanged,
including removal/remapping of already accepted clusters/member links from a fallback event. Full-length NumPy
pulse recovery is unchanged. Empty public finder and chain calls preserve their baseline outputs.

## Motivation and measurement limits

The supplied rcab1 profile attributed roughly **34 s to coincidence host dispatch versus about 1 s to packing /
splitting**. These are host dispatch/collection measurements, **not true kernel-only counters**. Existing
`dispatch_s`, `kernel_s`, `pulse_collect_s`, and `coinc_finish_s` can include compilation, queuing, synchronization,
transfer, host assembly, or recovery work; their names do not isolate GPU execution. In particular `kernel_s`
times collection and host `take` work, not a separately synchronized kernel-only interval. This patch does not
reinterpret or rename those counters.

No GPU timing, speedup, full-residency, or physics-validation claim follows from the local CPU regressions.
Power-of-two row buckets add bounded compilation variants and device slices may themselves introduce dispatch
work. Before promotion, measure baseline and opt-in on the same GPU/input/configuration, separate cold compile
from warmed end-to-end runs, synchronize completion, compare exact memberships plus all output properties, and
record tier occupancy, promotions, fallback counts, memory use and source/backend provenance. Validate supported
oneAPI devices separately; the local test checks policy, not the oneAPI compiler/runtime.

## Executed local regressions

Environment: `/Users/rmina/mu2e/.venv/bin/python3`, Python 3.14.5, JAX 0.11.2 CPU. Imports were verified to resolve
to the owned `core_e2e_gpu13` and `e2e_gpu13` clones. No remote calls, installs, builds, or git writes were used.

```bash
cd /Users/rmina/mu2e/remote/analysis/crv_wt/e2e_gpu13
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/Users/rmina/mu2e/remote/analysis/crv_wt/core_e2e_gpu13:$PWD \
JAX_PLATFORMS=cpu CRVFASTSIM_COMPACT_DISPATCH=1 \
/Users/rmina/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_reco_dispatch.py tests/test_coinc_attachment_jax.py \
  tests/test_chain_reco_contract.py tests/test_reco_jax.py tests/test_chain.py \
  tests/test_background.py tests/test_runconfig.py
```

Results: **114 passed in 66.32 s** with compact mode enabled; **114 passed in 93.43 s** with the environment
variable unset (same command, omit `CRVFASTSIM_COMPACT_DISPATCH=1`). These suite durations are not a performance
comparison. The new dispatch module contributes **38 tests** and explicitly runs both baseline and compact paths. Coverage includes exact array values/dtypes, explicit expected
member sets, attachment on/off, positions requested/omitted, full and partial batches, sparse event/member IDs,
power-of-two and non-power-of-two caps, promotions, residual overflow and oversized-group fallback, split groups
with whole-event replacement, fallback disabled, empty inputs, nonidentity waveform selections, all pulse recovery
flags, and chunk counts 1/3/8. Instrumented tests verify pre-transfer slicing, one whole-pytree transfer per
collection, production-tier row-capacity selection, and captured mode across asynchronous collection/promotion.
Real-kernel cases use small synthetic tier caps to keep CPU tests bounded; the production-capacity shape test
instruments the kernel launch rather than allocating a giant baseline tier. Membership and property comparisons
between baseline and compact use exact assertions, not relaxed floating-point tolerances.

## Real-array follow-up

Production GPU batch shapes can alter floating-point reduction ordering even when synthetic tests are exact.
Same-backend raw-array fingerprints must therefore be checked before interpreting a speed result. The benchmark
comparison driver reports exact identity separately from an explicitly requested bounded-property diagnostic.
Only cluster PE, average time, position and slope may use its `1e-8` absolute tolerance; pulse arrays, membership,
cluster identities, positions-calculated flags, start/end times, cardinalities and dtypes remain exact. This
bound is a regression diagnostic, not a physics-validation replacement. Run stock-pulse V-R3 for each new backend,
keep overflow/fallback enabled, and retain the default-off switch until whole-chain validation is complete.

