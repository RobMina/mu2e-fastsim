# Opt-in device transport

`transport` and `runchain` accept `--transport-backend jax`; NumPy remains the
default. `e2e-bench` accepts the same option. JAX is an optional dependency and
is imported only when requested. Stages A and C are supported. Single-scattering
tails, sliced MSC and all below-plane world/MSC variants raise rather than falling
back. This is not certification for the named Run 1A/Run 2 worlds.

```bash
python3 -m crvfastsim transport gen.root --stage C --gdml world.gdml \
  --transport-backend jax --outdir device_steps
python3 -m crvfastsim runchain gen.root --gdml world.gdml --calib calib.csv \
  --jax --transport-backend jax --outdir device_chain
```

## Numeric contract

`crossings_jax` does slab clipping and stable bounded top-k entry-distance
ordering on the selected JAX device; equal distances retain ascending bar index. A bounded number of crossings per track is retained (64 by
default); overflow raises before any table is published. Batches are padded to
4096 muons (the warm timing minimum of the 2048/4096/8192 sweep). A cheap host prefilter admits the union of the NumPy near-plane mask
and original-ray detector-box hits; unselected tracks can neither be scattered
by the default nor enter a bar. It is not a new acceptance cut.

`transport_jax.DeviceTransport` prepares geometry/world constants once. Convex
pieces use the same clipping and bounding-box face convention as core. Materials
are processed in the NumPy world's order. Missing material constants raise in
this backend instead of using NumPy's concrete fallback. World slabs use core's
slice, tail-overloss and stopping rules. Above-plane scattering uses the default
band covariance and geometric-mean momentum. A rank scan evaluates each bar at
its entry momentum, draws one loss, caps it at kinetic energy and shortens the
reported path at a stop. It intentionally preserves the NumPy exit-position and
generated-beta clock conventions; it does not repair or change these physics.

The CRV-local `eloss_transport_jax` execution twin reuses core material, dE/dx,
kinematic and fluctuation helpers. It evaluates independent Poisson/Gamma random
rounds in parallel and unrolls the bounded Poisson CDF recurrence instead of
launching a nested 32-round GPU loop for every transport slice. The K=32 cap,
ten Gamma rounds, first-accepted Gamma fallback and per-round key substreams
remain unchanged. Parallel ion summation changes float64 rounding, so agreement
with the core sampler is tested numerically rather than promised bit-identical.
No sampling threshold, distribution formula or transport stop rule is changed.

Seeds use an explicit JAX key derived from one draw of the caller's NumPy RNG.
Identical inputs, key, device, batch size and capacities repeat; NumPy and JAX
seeds are **not paired random draws**. Changing batch size changes the random
stream, not the target law. Empty/zero-length and padded lanes must not influence
loss ledgers or overflow. Full primary truth identities, including multiple muons
in an event, are retained.

For the new extracted range_out model, see [device-new-default.md](device-new-default.md).
The legacy certificate and measurements below do not certify that model.

Numeric batches stay on device through world loss, scattering, crossings and bar
loss. At the tier-1b boundary they are materialised into ordinary pandas CrvStep
and particle tables with the unchanged schema. Tier-1b stays on the host and the
existing response stages perform their usual transfers. This is **not** an
all-device end-to-end chain or a zero-copy handoff.

## Reproducible checks

`python3 -m valhist.device_transport --help` describes the held-out table receipt,
exact deterministic crossings, event-clustered step histograms, independent and
same-event paired yield errors, V1 stock comparison, and synchronized cold/warm
wall timing. The driver uses existing inputs in place and writes summaries and
candidate tables only under its explicit output directory. Cold means first call
in-process, including compilation; warm reuses prepared geometry. Runtime is not
extrapolated from smaller jobs.

Synthetic tests cover slab faces, overflow, schema, padding, multi-muon identity,
repeatability, stops, empty inputs, bypass populations, convex columns, unsupported
options and the slab-loss law. Run tests on a compute node, not a login node.

See [validation and timing details](device-transport-performance.md) for the
measured boundary, failed experiments and reproduction tools.

## Measured transport result

On one A100-80GB, ten 50k-generated-event held-out shards, five synchronized warm
JAX repeats per shard and four warmed NumPy repeats: end-to-end transport median
0.230–0.302 s versus 0.677–0.904 s, a per-shard 2.68–3.14x speedup. This includes
host preparation, transfers and table assembly, excludes one-time compilation,
and is not a whole-chain or whole-node rate. Cold compilation remains about two
minutes, so short one-shot invocations can lose overall.

The fixed 100-even-shard, 5M-generated-event bank gives 1,331,412 JAX steps versus
1,330,959 NumPy, +0.0340% ± 0.0466% same-event clustered error. Deterministic
crossings and replay agree; all 21 aggregate step-shape flags are false. Stock
V1 and JAX-versus-NumPy V1 pass in EX/T1/T2/M without changed thresholds. The
optimized loss sampler passes the original 14-point, 4M-sample-per-arm checks.
This certifies the extracted default transport, not reserved physics variants.

A separate 500k-generated-event audited chain is essentially tied in warm wall
(88.221 s NumPy versus 88.236 s JAX): transport improves 6.750 to 2.752 s while
independent downstream shower/reco work offsets that gain. Do not infer a full
chain speedup from the transport result. With existing prefetch enabled for both
backends, a further audited run observes medians 62.930 versus 60.194 s, with
overlapping three-repeat timing ranges; JAX outputs are exactly unchanged by
prefetch. Treat this modest net gain separately from the robust transport gain.
Tier-1b and reco remain unchanged.

## Tier-1b device design (not implemented)

The current library is pandas-indexed and has variable donor histories. A future
port should build immutable device arrays at preparation: donor offsets/lengths,
per-(group, momentum, path, context) support indices, complete truth ancestry,
geometry placement maps and sidecar data. Sampling keys must be per generated
primary/crossing and separate from transport. The public sampling options and
returned CrvStep schema stay the boundary; overflow or missing support must fail
closed, not truncate histories or invent fallback donors.

The evolving whole-muon/stop-product formats need their own validated CSR offset
contracts before device packing. Sampling and placement should remain separate
kernels, with containment and event/muon correlation checks against the host
implementation. Budget both donor-step multiplicity and per-event response work;
a fixed maximum taken from one training shard is not safe. No sampler changes are
part of this transport backend.
