# Device-resident execution backend

`device_backend.Backend` implements resident source sampling, joint overlay and a
compiled response-kernel interface. `device_physics:model` now connects **physical
tracker digitisation and onspill CRV response through coincidences**, with all
intermediate packing on device. The original synthetic arithmetic model remains
available for infrastructure tests. Existing staged consumers and shared tracker/CRV
kernels are unchanged; `stream_backend.Backend` is still the default. Tracker track
reconstruction is not implemented here.

See [DEVICE_TRACKER.md](DEVICE_TRACKER.md) and [DEVICE_CRV.md](DEVICE_CRV.md) for
physical scope, explicit capacities and implementation-regression tests. Production
physics closure, dense-memory qualification and sustained large-job throughput are not
established by the small integrations.

## Execution boundary

Setup validates five complete frame pools and constructs flat CSR columns once,
then uploads pools, assets, seeds and donor state to one explicitly selected JAX
device. Host source snapshots also remain resident. Each batch uploads only three
scalars (ordinal, batch index, event count). One JIT entry point performs:

1. Beam-weighted truncated lognormal intensity and four conditional Poisson draws.
2. Sequential wrapping donors, inclusive initial skips once, persistent cursors.
3. Joint truth/tracker/CRV overlay, stable source order and lossless particle,
   parent and generator reference remapping.
4. The supplied array-only response kernel, with separate tracker/CRV RNG domains.

Intermediate arrays remain device arrays. Nested host callbacks are rejected from
the traced program; implicit transfers are disallowed during dispatch. There is
one explicit terminal `jax.device_get` call for output, sampling receipt and errors
(not necessarily one hardware DMA operation). Host code only compacts final output,
serializes sampling/ROOT files and validates persisted entries. Setup, Python batch
control, scalar dispatch, synchronization and filesystem I/O are not on the GPU.
XLA implementation-internal scheduling is not certified by this Python-level test.

`JAX_ENABLE_X64=1` is required. Full original uint64 identities, including values
above 2**63, are distinct from contiguous realization ordinals. RNG uses JAX keys
with both halves of uint64 seed and batch index folded in; it is not NumPy/stock
byte replay. Changing batch shape changes the realization. Finite source reuse
still does not supply new independent physics events.

## Bounded buffers and failure behavior

Sizes measured for the Run 1A mixed configuration are in [capacities/](capacities/README.md).

A JSON capacity file explicitly sets `max_donors` per event per stream and `rows`
for every collection. The optional `response` object is required by the physical
factory and contains separate `tracker` and `crv` capacity dictionaries. The compiled graph uses fixed shapes and masks, including
for a partial final batch. Donor, collection, aggregate-row and response overflow
flags reject the entire batch before its ROOT/sampling output or cursor commit.
There is no accepted truncation and no NumPy fallback. A failed run has no successful
summary. Retrying a write failure needs a fresh output directory; existing output
files are never overwritten by the backend.

The shapes bound expanded arrays, not total memory: resident host/device pools,
receipts, JAX executables and workspaces also cost memory. Capacities have not been
qualified for dense production pileup. Choose them explicitly, not from this tiny test.

## Response factory contract

`--backend mixing.device_backend:Backend` selects its own `Source` automatically.
`--device-kernel module:factory` is mandatory; no physical kernel is silently chosen.
Use `mixing.device_physics:model` with the existing manifest/timing/conditions/optical
assets for physical processing. This factory validates full-ID timing at setup,
gathers it by device signal indices, and joins CRV PDGs by remapped particle identity
on device. It emits the familiar tracker d_sid/d_tdc0/1/d_tot0/1/d_pmp/d_flags/d_nadc/
d_adc schema, not an incompatible flattened TDC-pair branch.
The factory is `factory(args, source, layout)` and returns an object with:

- `kind`: `synthetic` or `physics`; synthetic requires `--synthetic`.
- `support`: implemented stage names. The arithmetic fixture advertises only
  `synthetic_tracker` and `synthetic_crv`; the physical factory advertises
  `tracker_digis` and `crv_reco`. Neither supplies `track_reco`.
- `metadata`: honest model provenance and `scope`.
- `assets`: a numeric-array pytree, uploaded once.
- `apply(batch, assets, crv_key, tracker_key) -> (output, error_flags)`: pure JAX,
  no callbacks, host compaction or data-dependent Python control flow.

Batch carries fixed-size `keys`, `event_mask`, `ordinal`, `size`, and per-collection
`columns`, `event`, `mask`. Invalid lanes must not contribute. Output has `tracker`,
`digis`, `pulses`, `clusters` tables, each with `mask` and dense `ev`, plus `timing`
(mc_pbtime/reco_pbtime/event_length). Tracker columns become jagged `d_*` branches;
ADCs are flattened per event. CRV digis use a fixed ADC width. Error flags are checked
before writing. Additional track-reconstruction event tables are not implemented
in this adapter; requesting `track_reco` is rejected.

## Occupancy output (opt-in)

`--device-output occupancy --occupancy-geometry GEOMETRY.json` (default `root`) keeps the
analysis output on the device too. Inside the same compiled batch, `device_occupancy.reduce`
turns the masked detector tables into dense per-event arrays: straw digis per tracker plane
(36), reconstructed CRV pulses per sector (bar map with the `counts.py --geometry` contract),
and CRV digi and coincidence-cluster counts. The terminal transfer then carries only these
arrays, the event identities, the error flags and a compact sampling receipt: per-donor
arrays are dropped because sequential donors follow from the per-stream counts and the
initial cursor. Each batch writes `sampling.{json,npz}` and `occupancy.npz`; no detector
digis, ADC samples or pulses are persisted, and `fast.root` is not written. The run's census
check verifies identities and realization ordinals from `occupancy.npz`.

`python3 -m mixing.device_occupancy RUN --stock counts.npz --selection TEXT --out NEW.npz`
pairs the measured batches of a finished run with saved stock counts by full event ID
(`counts.align`). Run exactly one pool population of measured events, so that each signal ID
appears once. `mixing.device_bench` wraps the same backend as entry points for the shared
warm-rate timer (`mu2efastsim.bench`). Tests: `test_mixing_device_occupancy.py`, and in
`test_mixing_device_physics.py` the same stream with both outputs must give identical counts.

## Pipelined output (bench default)

Serially, each batch is dispatched, waited for, downloaded, checked and written before the next one is dispatched,
so the host transfer and the ROOT write sit between device steps. `--device-pipeline` (stream benchmark; off by
default there) or `"device_pipeline"` (bench configuration; **on by default**, `false` for the serial loop) uses `Backend.dispatch(request)` and `Backend.collect(handle, dir)`:
- batch k is downloaded, checked and written on one worker thread while the main thread dispatches batch k + 1.
  A GPU execution holds its calling thread (XLA reads while-loop predicates back), so the overlap needs the thread;
- the source and the device state advance at dispatch, so the batches, keys and outputs are those of the serial
  loop;
- a rejected batch still raises, in `collect`, and stops the run;
- two batches' outputs are on the device at once.

`--device-pipeline-writers W` (stream benchmark) or `"device_pipeline_writers": W` (bench configuration; default 1)
collects up to W batches at once, each on its own worker thread: batch k + W is dispatched once batch k is written.
Each batch writes its own files, so the files are those of one writer; ledger rows stay in dispatch order. The ROOT
compression (zlib) runs outside the Python lock, so writers overlap the write, which with full digis takes longer than
the device step once that is under about 0.35 s per batch of 256. W + 1 batches' outputs are on the device at once.

Per-batch metrics then report `wait_transfer_and_write_s` instead of separate device and transfer times.

## Output compression (opt-in)

ROOT output uses uproot's default compression (ZLIB level 1), and the benchmark tree is added in a second open of the
file. `--output-compression zstd1|lz4|none` (stream benchmark) or `"output_compression"` (bench configuration) writes
every tree in one open with that compression: the decoded trees and branches are the same, but the file bytes differ.
On one real batch of 256 mixed events with full digis, the write takes 0.25 s with ZLIB 1 (5.4 MB), 0.10 s with
ZSTD 1 (4.9 MB), 0.08 s with LZ4 (7.1 MB) and 0.05 s uncompressed (15.8 MB).

## Reproducible synthetic integration

From the umbrella with JAX/uproot/awkward dependencies installed:

```bash
JAX_ENABLE_X64=1 python3 -m mixing.device_synthetic --platform gpu --out NEW_PATH
```

For CPU development use `--platform cpu` explicitly; GPU selection never silently
falls back. The fixture creates five sparse pools including empty events, parents,
reused donors and IDs above 2**63. Synthetic response is channel-wise energy summing,
random arithmetic, masked ADCs and fake pulses/clusters—not detector physics.
The runner writes four warm-up events, then 13 measured realizations in batches
4/4/4/1 and rereads every finished census and ordinal. Its numerical rate is only a
fixture measurement and must not enter production comparisons or slides.

## Validation completed 2026-11-06

- Local actual-JAX tests compare every valid overlaid column and event assignment
  against the existing independent host overlay using the device-selected donors.
- Coverage includes repeat-output equality, continuation, all-empty events, partial
  tails, >2**53 ordinals, signed/unsigned sparse keys/channels, rejected capacities,
  no state commit on failure, nested callback rejection and physical-stage refusal.
- An additional test loads real tiny ROOT sources through the default device source
  factory and runs the complete persistent runner. No GPU mock is used in these tests;
  their execution device is CPU.
- GPU arithmetic integration: **exit 0**, NVIDIA RTX A6000, two host CPUs,
  16 GiB request. Real CUDA graph execution, four warm-up plus 13 measured
  entries, all finished outputs verified. Whole subprocess 17.036 s.

Physical response was subsequently ported; the arithmetic test alone does not
validate that port. Reconstruction, physical closure, representative dense memory
behavior and production-scale throughput remain **not validated**.

## Physical-kernel integration

`mixing.device_integration` runs the physical adapters on artificial joint sources,
with an existing onspill tracker conditions NPZ and the stock CRV single-PE template.
CRV geometry/calibration/optical probabilities are explicitly synthetic; tracker
steps are placed using the loaded geometry. It is not a detector-accuracy sample.

```bash
JAX_ENABLE_X64=1 python3 -m mixing.device_integration --platform gpu --out NEW_PATH \
  --tracker-conditions SAVED_CONDITIONS.npz --template singlePEWaveform_v3.txt
```

Default: two warm-up events, five measured (2/2/1), all completed ROOT censuses
verified, all four detector outputs required nonempty across the run. An extra
same-device physical tracker check compares against the old staged driver outside
the measured interval. `integration_check.json` is written only if this succeeds.
The fixture's numerical rate is not a production throughput result.

Physical integration result (2026-11-06): **exit 0**, RTX A6000 (49,140 MiB),
two CPUs/32 GiB requested. Seven persisted events including warm-up:
**9 tracker digis, 95 CRV digis, 95 pulses, 6 clusters**. Same-device tracker
reference: five events/three digis, all integer columns exact, maximum analog-time
difference **0**. Whole subprocess 241.710 s includes compilation, all outputs
and the extra staged comparison. No production-rate interpretation is attached.

An initial GPU attempt stopped during compilation when the transfer guard caught
captured tracker condition arrays being downloaded, before output. Explicit operand
lifting fixed that without relaxing the guard. The malformed-PDG guard and its regression
tests postdate the successful GPU snapshot; valid integration PDGs are unchanged.
No shared detector kernels or shared tracker files were edited by the device port. The separate
CRV onspill dependency supplies the opt-in timing, waveform gate and thermal hooks.
