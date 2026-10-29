# Indexed per-input raw replay cache

`crvfastsim.veto.indexed_replay` is diagnostic veto-closure infrastructure. It reads **one
local retained replay input per cache**, without optics, response, stock reruns,
or changes to the frozen full registry or retained plan. It does not admit a
full retained-sample execution or certify independent exposure.

## API and binding

```python
from crvfastsim.veto.indexed_replay import Limits, load_input

# These digests must come from the already admitted, saved bindings.
with load_input(
    raw_path,
    full_registry,
    expected_registry_sha256=saved_registry_sha256,
    expected_input_sha256=saved_input_sha256,
    expected_manifest_sha256=saved_manifest_sha256,
    limits=Limits(),
) as cache:
    selected = full_registry[
        (full_registry.input_sha256 == saved_input_sha256)
        & full_registry.frame_id.isin(batch_frame_ids)
    ]
    batch = cache.select(
        selected, expected_registry_sha256=saved_registry_sha256
    )
    events, steps, particles = batch.events, batch.steps, batch.particles
```

Always pass the complete, canonical frozen registry to `load_input`; never
freeze a subset to regenerate labels. Its digest must match the externally saved
full-registry digest. The entire input's `dump/events` domain must match the
registry's rows for that input, including exact saved `pbtime` and `intensity`.
The cache does not retain or mutate the caller's registry. At selection, supply
exact rows with the registry's original columns, full tuple and `frame_id`.
Duplicate frames, foreign inputs, changed labels, changed exposure, and a changed
expected registry digest fail. Selection order is arbitrary, but all selected
frames must belong to this one input. An empty selection is allowed.

`load_input` requires lowercase SHA256 bindings for the ROOT file, its `.json`
sidecar, and the full registry. ROOT and sidecar hashes are checked both before
opening ROOT payloads and after reading/validation. No cache is returned on a
post-read mismatch. These checks are not a filesystem lock or a proof against
concurrent changes that are restored between checks; use immutable local inputs.
Selections reuse the admitted in-memory snapshot and do not rehash or reopen files.

The current format is deliberately the **legacy raw replay-v1** contract used by
`optical_timing_probe.read_input`: `raw_times=true`, matching manifest/tree counts,
and a nonempty all-event table with the declared single run/subrun and consecutive
event numbers beginning at `first_event`. This does not generalize that manifest
to multi-run or multi-subrun inputs. Joins nevertheless use all `(run, subrun,
event)` fields, and outputs additionally carry the input SHA256. Replay-v2 is
rejected rather than silently discarding its exposure contract. The sidecar must
be a JSON object with `schema`, `raw_times`, `output_sha256`, `entries`, `events`,
`run`, `subrun`, and `first_event`. Schema and digest are strings, `raw_times` is
a boolean, and counts/identities are nonnegative JSON integers (not booleans or
floats). `entries` must contain exactly `events`, `steps`, and `particles` counts.
Missing or mistyped fields fail with `ValueError` before legacy-schema checks
or ROOT opening.

## Reading, indexing and output

The loader requires flat TTrees with exactly the columns and dtypes from replay's
`EVENT_TYPES`, `STEP_TYPES`, and `PART_TYPES`. It scans each tree once in bounded
row chunks, filling preallocated structured NumPy arrays in ROOT source order.
There are no repeated per-frame min/max ranged reads, concatenated chunk caches,
or per-event Python loops. Stable vectorized sorting plus per-event offsets index
both steps and particles. A vectorized full-tuple particle lookup is constructed
once and reused for arbitrary whole-frame selections.

Validation covers the complete input, not just the selected frames:

- unique all-event identities and exact registry domain/exposure equality;
- finite event phase/intensity and nonnegative intensity;
- steps and particles entirely inside the all-event census;
- positive replay particle keys, nonzero PDGs, and replay's synthetic
  `parent=-1`, `isprim=1` convention;
- unique `(run, subrun, event, key)` particles and a matching particle for every
  step's `(run, subrun, event, sp)`;
- nonnegative bar/energy, positive replay `sp`, finite non-time step columns,
  and no infinite endpoints. NaN endpoints are preserved, not filtered.

Unused valid particles are allowed and returned with their frame. Particle keys
need not be contiguous. No detector geometry, channel-status or optical validity
is inferred by this raw loader.

`Selection` contains `events`, `steps`, `particles`, `registry_sha256`,
`input_sha256`, and `typed_payload_bytes`. All three tables carry
`input_sha256`, `run`, `subrun`, `event`, `frame_id`, and **original `source_row`**
(the zero-based row in that table's ROOT TTree). Steps additionally carry exactly
joined `pdg` as int64, matching the probe's joined-column convention. Replay input
columns keep their declared types and raw values. Events follow requested frame
order; steps and particles follow original source order across the whole
selection, even when frames are interleaved or requested in reverse order. Empty
frames remain in `events`, and empty step/particle tables retain typed columns.
No frame is split or silently dropped. Repeating a selection returns new owned
tables; caller edits do not change the cache.

## Resource admission and errors

`Limits` has these defaults:

| Limit | Default | Meaning |
|---|---:|---|
| `max_events` | 1,000,000 | Input all-event rows |
| `max_steps` | 2,000,000 | Input raw step rows |
| `max_particles` | 2,000,000 | Input particle rows |
| `max_resident_bytes` | 512 MiB | Exact persistent NumPy array payload |
| `chunk_rows` | 32,768 | Requested rows per ROOT iteration chunk |
| `max_basket_bytes` | 16 MiB | Maximum uncompressed bytes per branch basket, including its TKey header |
| `max_selection_bytes` | 512 MiB | Pre-gather selected typed-payload budget |

Limits must be positive built-in integers at most `2**52`. Input row and resident
byte checks happen using tree entry counts **before any payload arrays are
allocated or read**. Branch dtype/shape and basket-header admission also precede
payload allocation. For each normal (free) basket, admission reads only
`branch.basket_key(index)` and checks **`key.fKeylen + key.fObjlen`**: TKey header
bytes plus uncompressed object bytes. Equality with the cap is allowed; this is
not a compressed-byte or data-only convention. An oversized basket fails before
any basket payload decoding/decompression rather than relying on uproot to split
it. In uproot 5.7.4, `basket_uncompressed_bytes()` calls `basket()` and decodes
payloads, so it is deliberately not used for admission.

Embedded baskets are unsupported and rejected without materializing them.
`minimal_ttree_metadata=True` keeps embedded baskets lazy at ROOT opening. The
loader checks uproot's private `_embedded_baskets` and `_num_normal_baskets`
metadata state, tested against uproot 5.7.4, rather than calling `num_baskets` or
`embedded_baskets` (which can materialize payloads). Missing/unknown metadata state
fails closed; compatibility with other uproot implementations/versions is not
certified. ROOT metadata itself may still be read/decompressed before admission;
this is not basket-payload decoding and is outside the byte cap. Chunk
length, dtype, and total scanned-row coverage are checked while reading. ROOT
array and object caches are disabled.

For event/step/particle counts `E`, `S`, `P`, `resident_bytes(counts)` reports

```
E * EVENT_DTYPE.itemsize + S * STEP_DTYPE.itemsize + P * PART_DTYPE.itemsize
+ 8 * (2*S + P + 3*E + 2)
```

The additional arrays are step and particle order indexes, two event-offset
arrays, one frame-ID array, and the step-to-particle lookup. The loader verifies
this predicted size against actual persistent `nbytes` before returning.
`cache.resident_array_bytes` exposes that value. Source rows are original array
positions, not a second resident identity table.

The separate selection budget is checked before row gathering or DataFrame
construction. It accounts for replay typed fields, int64 `source_row` and
`frame_id`, a fixed 256-byte Unicode SHA256 slot per output row, and an additional
int64 joined PDG per step. It is a **typed serialization/payload convention**, not
the actual pandas string representation or a pandas memory estimate.

**Neither byte cap bounds peak RSS.** Registry normalization, pandas indexes,
validation arrays, sort/gather temporaries, Python strings, ROOT metadata,
compressed buffers, decompression, concurrent basket work and DataFrame copies
are outside persistent `nbytes`. `chunk_rows` does not split a ROOT basket. The
per-basket cap is not a total decompression-memory cap. Multiple retained
selections or multiple independently created caches are caller-owned aggregate
memory; this module has no process-global accounting. Use one context at a time,
release batch outputs promptly, and measure peak memory separately before scaling.
`close()` releases cache arrays and prevents later selection; existing outputs
remain caller-owned.

Malformed contracts, identities, domains, joins and cap overruns raise
`ValueError`; filesystem and uproot decoding errors propagate. A failure never
truncates output or renormalizes the denominator. No disk output or completion
receipt is created by this module, and `cache.binding['execution_admitted']` is
always false. Photon/cascade caps, waveform preallocation, output-byte accounting,
atomic completion, real-asset downstream validation and the pulse-fit
partition-invariance gate remain separate requirements. Response replicas on this
registry are not independent exposure.

## Isolated validation

In an environment with the project's Python dependencies (including its core
dependency) installed or on `PYTHONPATH`, run only this file from the project
directory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q -p no:cacheprovider \
  tests/test_veto_indexed_replay.py
```

Tests use synthetic fake trees and tiny locally created real uproot TTrees. They
exercise interleaved rows, source order, full-tuple identity and particle joins,
empty frames and stepless inputs, repeated/partitioned selection, malformed
particle and step domains, changed full/selected registries, branch contracts,
manifest semantics, before/after input hashes, one scan per tree, and rejection
of caps before payload allocation or selection gathering. Real-uproot spies cover
basket decoding, decompression and interpretation: cap rejection performs none,
and successful header admission adds no payload decode/read before the one scan.
Embedded-basket lazy-state tests reject without invoking its materializing accessor.
No retained real data, completed full controls or full test suite is read or rerun. These tests establish
a loader contract only, not real-sample runtime, peak memory or execution admission.
