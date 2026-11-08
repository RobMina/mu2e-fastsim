# Persistent end-to-end benchmark infrastructure

This is infrastructure for a future production-scale measurement, not a new speed
claim. The 2,741-event process benchmark is preserved as a small-job latency measurement.

## What runs once and what runs per batch

- `StreamSource` loads and validates the signal and four complete joint donor pools
  once. They remain resident. Each batch samples a common intensity and four Poisson
  multiplicities, continues sequential donor cursors and overlays before digitization.
- The default `stream_backend.Backend` loads geometry, timing, conditions and device
  tables once. It calls the existing tracker digitizer and on-spill CRV chain without
  changing kernels, precision, model parameters or device-chain optimization flags.
- `stream_benchmark` performs explicit warm-up batches, then measures a requested
  number of realizations, including a smaller final batch if necessary. It retains
  no growing list of detector payloads. Each batch writes a closed ROOT file and its
  sampling record; an append-only JSONL ledger records batch timings.
- After the measured interval, the runner checks highest-cycle persisted event
  censuses, tracker/CRV identities and realization ordinals. Failed/partial runs retain
  their files but do not publish a successful rate.

The default backend is **GPU-backed with host preparation, packing and intermediates**,
not an entirely on-device chain. Transport, calorimeter, trigger and tracker track
reconstruction are absent. No dummy reconstruction stage or placeholder throughput is
reported. Device synchronization is required at interval boundaries.

## Identity, randomness and finite-pool reuse

A realization is identified by its monotonically increasing `benchmark/events.realization`
ordinal plus the original full integer run/subrun/event. Signal positions wrap through
one resident finite pool. A batch cannot exceed that pool's population, so its original
IDs remain unique. Donor cursors continue across batches; starting skips are drawn once.

Sampling uses an explicit seed and batch-domain RNG, not the old single-manifest RNG
sequence. Detector seeds likewise vary by batch. Repeating the same invocation is
reproducible; changing batch size, detector chunk size or warm-up count is a different
realization. Warm-up advances the source state and has its own output identity range.
This is a repeated finite-pool workload, **not new independent primary physics events**.
It is appropriate for exercising throughput, not for reducing statistical uncertainty
by counting repeated source events as independent. Saved source timing is reused.

## Memory and output limits

`--batch-events` bounds the number of events assembled at once. `--max-selected-rows`
rejects an oversized selection before expanded truth/step overlay, without truncation.
The row limit is not a byte-level memory guarantee: all source pools are resident,
selection arrays and framework/device workspaces also consume memory. The default
backend still uses its own `--chunk` response batches. These two controls are distinct.

Every warm-up and measured batch is written. Output volume grows with the requested
workload; choose disk allocation explicitly before submitting. File close means written
to the filesystem, not forced durable storage (`fsync` on payloads is not requested).
No production-scale memory/storage qualification has been performed. The separate
device backend has passed a tiny synthetic CUDA integration, not physical-response
validation; see [DEVICE_BACKEND.md](DEVICE_BACKEND.md).

## Timing contract

`stream_report.json` separates:

1. `setup_s`: source loading, backend construction, configuration and synchronization.
2. `warmup_s`: actual warm-up batches, including their output.
3. `measured_s`: selection, overlay, detector response, synchronization, sampling/ROOT
   output and ledger writes for the measured population. Rate uses **verified finished
   entries / this elapsed interval**. It does not assert that one warm-up eliminates
   all compilation; remaining compilation is paid inside the measured interval.
4. `pipeline_s`: setup through completion of all batch writes; not interpreter startup.
5. `validation_s`: post-run persisted census checks, outside the sustained-rate timer.

For actual whole-job cost, wrap the CLI with `mixing.process_timer`. Its interval
includes interpreter startup, setup, warm-up, writing, readback and shutdown. A
container launcher can also record container-launch wall time, CPU usage and peak RSS. Neither
sum stage medians nor multiply the small-job rate to forecast a campaign runtime.
No stock baseline or speedup is invented by the runner; compare to a separately
measured large-job stock boundary and resource allocation.

## Invocation

All paths refer to existing validated inputs; no source extraction or build is done.

```bash
python3 -m mixing.process_timer --receipt NEW_PROCESS_TIME.json -- \
  python3 -m mixing.stream_benchmark \
  --frames-manifest SOURCE/direct_manifest.json --expanded RESPONSE/expanded.fcl \
  --truth MC_TIMING.root --stock RECO_TIMING.root --photon-csv CRVPhoton.csv \
  --tracker-conditions conditions.npz --out NEW_DIRECTORY \
  --events N --batch-events B --max-selected-rows ROW_LIMIT \
  --warmup-batches W --chunk C
```

An allocation-side launcher can wrap this command in the existing analysis image,
record the actual hardware and private compilation-cache state, and use the external
process timer above. No physical production streaming campaign has been measured.

## Future backend interface

`--backend module:factory` constructs one backend as `factory(args, source)`.
It must provide:

- `metadata`: JSON-compatible `name`, `support` stage names, honest `residency`, and
  optional `output_event_tables` for additional stages' empty-inclusive event censuses.
- `run(batch, output_directory)`: process the batch, write `fast.root` with the five
  existing detector tables and `benchmark/events` original IDs/realization ordinals,
  return a small metrics dictionary containing `events` (no retained payload arrays).
- `sync()`: wait for outstanding device work before the timing boundary.

The default stages are `tracker_digis` and `crv_reco`. Future track reconstruction should
advertise `track_reco` only when implemented, persist an event census including failed
and zero-track events, and list that table in `output_event_tables`. Requesting
`--require-stage track_reco` currently fails explicitly. Numerical convergence of
individual fit components is not full reconstruction validation.

The default source batch contains host detector inputs plus original identity/provenance.
`device_backend.Backend` selects its own request-only source automatically and runs
device sampling/overlay with an explicit pure-JAX response factory. The arithmetic
fixture requires `--synthetic` and cannot satisfy physical stage requests. The
`mixing.device_physics:model` factory provides physical tracker digis and onspill CRV reco,
with `response.tracker`/`response.crv` capacities in the device capacity JSON. See
[DEVICE_BACKEND.md](DEVICE_BACKEND.md) for interfaces and tests. With
`--device-output occupancy` the device backend reduces each batch to per-event plane and
sector counts on the device and writes `occupancy.npz` instead of `fast.root`; the run's
census check reads those files. Physical track reconstruction remains absent; the shared
tracker source and the old exact single-realization regression are unchanged.
