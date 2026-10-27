# Named-run cosmic time-window provenance

`runchain` uses the `crvfastsim.timewindow` helpers to make its existing cosmic window policy explicit,
saves the contract in stats and emit records, and validates compatibility before merging. Physics windows,
`runconfig.get`, event times, generator weights and denominators retain their existing defaults.

## Resolution and integration

```python
from crvfastsim.timewindow import resolve_runchain_window, validate_runchain_merge

rc, contract = resolve_runchain_window(run_config, window)
# Preserve the existing runchain special case for the extracted chain configuration.
# Build the chain configuration using rc, and check its actual window before saving.
assert list(cfg.window) == contract["effective_window_ns"]
metadata = {"run_config": rc.name, "window": list(cfg.window), "time_window": contract}
res.update(metadata)
emit_record.update(metadata)

# Call before summing counts, combining sector names, or writing merge output.
merge_metadata = validate_runchain_merge(shard_records)
merged_result.update(merge_metadata)
```

`resolve_runchain_window(run_config=None, window=None)` returns a resolved
`RunConfig` and a fresh JSON-safe `crv-runchain-time-window-v1` record. The
configured window is the unmodified named-run default; the effective window is
what this cosmic run should actually digitise. An omitted override selects
**offspill, 0–100000 ns**, including Run1A, Run1B and Run2, whose configured
onspill defaults are **200–1500 ns**. An explicit `offspill`, `onspill`, or finite,
strictly increasing numeric pair overrides this policy. Invalid names, malformed
bounds, reversed/empty windows, booleans and nonfinite values fail with
`ValueError`. The registry remains unchanged.

The record includes the canonical run name, both windows, explicit/implicit
selection provenance, unchanged input time reference, zero rebasing, no generator
reweighting, and warnings. Warnings are returned as data, not emitted through
Python's warning machinery; callers should display them and retain them in JSON.
Equivalent explicit and implicit selections are retained separately on merging.

## Generator convention and evidence limits

The declared cosmic convention is `cosmicTimeOffset`, with a uniform offset interval **0–100005 ns**
(`Production/JobConfig/cosmic/prolog.fcl`, also present in the persisted named-reference process configuration).
This is commonly called the 100 us offset but is 5 ns longer than the 0–100000 ns offspill readout window. This is a **convention assumed by this workflow**, not a
measurement or validation of input generator times. In particular, an offset's
nominal window is not a bound on every absolute generator time, transported step
arrival, secondary, photon, or pulse time. Flight and secondary delays matter.
No event is rebased to the start of the digitisation window.

An explicit onspill or custom window changes time acceptance. It does not convert
the cosmic sample to a beam sample, reweight its generator, alter the generated
event denominator, or establish a retained-event fraction from a width ratio.
Matching the named configuration's nominal onspill window does not remove this
warning. Output provenance is a declaration, not input-file or physics validation.

### Onspill alignment is not implemented

Stock photon, SiPM and waveform producers use `eventWindowStart = -ProtonBunchTimeMC.pbtime_`.
They add it to the configured onspill gates and to each FEB's TDC0 phase. Stock waveform gates are therefore
about 400–425 ns to 1700–1725 ns in absolute time, not a fixed 200–1500 ns crop. `runchain --window onspill`
retains its historical fixed crop and explicitly labels that limitation; it is **not stock-onspill closure**.
The named reference FCLs use `SpillType: 0`, where stock starts at zero with a 100000 ns event length.

## Merge contract

`validate_runchain_merge(records)` consumes nonempty mappings already read from
JSON, raises `ValueError` before accumulation on incompatibility, and returns
JSON-safe metadata. It never mutates its inputs or writes files.

For new records, every shard must contain `time_window`, `run_config`, and
`window`. The latter two must agree with the nested contract. The schema,
generator convention, no-rebase/no-reweight policy, bound shapes and selection
provenance are validated. Run name, configured/effective windows and time
convention must agree across shards. `type_names`, when present, must be present
on every shard and agree after normalising integer/string sector keys. This
prevents the old last-shard-wins sector-name overwrite. Compatible selection
provenance and warnings are retained in the merged contract. Success reports
`time_window_status: declared_compatible`, **not** verified physics or verified
input support. The returned metadata can accompany a subsequent merged record.

New and legacy records cannot be mixed; a null/malformed/unknown-schema contract
is not legacy. Homogeneous legacy records (no `time_window` key anywhere) are
allowed only when all available run names, windows and sector mappings agree,
including field presence. Their metadata reports
`time_window_status: unverified_legacy` and a warning. Missing information stays
missing: the helper does not invent offspill provenance from today's defaults or
silently upgrade old shards. Empty inputs and partially declared legacy fields
fail. Legacy records already marked unverified remain unverified on later merges.

This is only a run/window and sector-label compatibility check. It does not
certify response settings, geometry overrides, source independence, duplicate
events, input hashes, generator support, reference-sample timing, or full physics
compatibility. Those require separate checks.

## Local tests

Use an explicitly pinned, read-only detector-independent core checkout and the
project interpreter, without bytecode or pytest cache writes:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$CORE:$PWD" \
"$PYTHON" -m pytest -p no:cacheprovider -q tests/test_timewindow.py
```

The integrated implementation was tested against a read-only core export at
`a35760693cf68d65d2a8b18f0a1a080b12cacd03`. Tests are synthetic and do not
require geometry, ROOT inputs, simulation, a build, or remote access.
