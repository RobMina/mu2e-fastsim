# Host stages in worker processes (`--host-workers N`)

On the JAX path, `runchain` runs two kinds of work per file:
- the transport (stage C) and tier 1b, in NumPy on the host;
- the response and reconstruction, on the device.

With `--prefetch` the next file's host stage runs in a thread, but the NumPy transport and tier 1b are Python-heavy and
hold the GIL against the main thread: a pass gets slower with two prefetch threads than with one.

`--host-workers N` (on `runchain` and `analyses.runchain_bench`; module `crvfastsim.host_workers`) runs the host stages
in N worker processes instead:

- All of a run's files are queued at the start; the main process takes their steps in file order.
- Each file keeps its own generator, `default_rng([seed, file index])`, and the workers run the same function as the
  in-process path (`host_workers.numpy_host_stage`). The steps, the particles and the generator state handed back are
  therefore the same: `runchain --save-reco` outputs are byte-identical with and without workers (float32 and float64).
- The workers receive the prepared context (geometry, world, absorbers, library; about 0.5 GB) once, when they start,
  and keep it across passes. They use JAX on the CPU only, for the transport's slab crossings.
- `runchain_bench` hands each pass the next pass's files (`host_ahead`), so the workers transport them while the
  current pass runs on the device, as they would in a stream of files.

Supported: table inputs with the NumPy transport and tier 1b, which covers the named run configurations. Not supported:
the native generator, step-table inputs, the detail benchmark, and the device transport or tier 1b.

## Device charge sort (capacity mode)

With `--capacities` the resident SiPM stage keeps the fired charges of every chunk on the device. They are written into
one buffer (capacity `charges`), sorted by (channel, time) there, and downloaded once per file. This replaces the
per-chunk downloads, the concatenation and a host `np.lexsort` of about 8 million charges per file. The order is the
same stable order, so the charges are the same arrays.
