# Transport history diagnostics (opt-in)

`transport(..., trace=observer)` copies explicitly selected source rows into
`begin`, `material` and `end` callbacks. The observer supplies distinct integer
`rows`; no RNG is supplied. `begin` records accepted field/boundary proposals,
`material` records sampler inputs/results, and `end` records updated states/fates.
Cold-at-start rows have terminal output but no accepted steps. The default
`trace=None` leaves this diagnostic inactive. A selection mask is computed once,
then indexed by the surviving source rows as transport compacts the batch.

The `tstrace.Trace(rows)` collector produces flat NumPy arrays with `step_` and
`material_` prefixes. The caller chooses rows, the full source batch, seed and
transport settings explicitly; there are no target or seed lists in the collector.
Keep the full source batch and its order when comparing stochastic histories:
simulating just the observed rows changes RNG consumption. For example, with
`source`, `field`, `geometry`, `rows` and `seed` supplied by the caller:

```python
from beamfastsim.tstrace import Trace
from beamfastsim.tstransport import transport

trace = Trace(rows)
result = transport(source, field, geometry, seed=seed, trace=trace)
columns = trace.arrays()
```

`tshistory.material_episodes(columns, row)` groups consecutive same-material
sampler calls, retaining call order, used column and bracketing states.
`tshistory.x_plane_samples(columns, row, planes)` returns every directional chord
intersection of caller-supplied x planes, ordered by step and within-step fraction.
Both require explicit row selection and accept in-memory or saved trace arrays.
Neither reads run records or launches transport.

## Recorded stock crossings

`tscrossings` selects exact `(run, subrun, event, track)` keys from the highest
cycles of `ntvd` and `nttvd` in explicitly supplied ntuples. Both `--ntuple` and
one or more `--id RUN SUBRUN EVENT TRACK` arguments are required, as is `--out`.
There is no implicit cohort, campaign path or species restriction. `ntpart` is
not read. In Python, use `extract(paths, targets)` or `select_exact(table, targets)`.

The NPZ retains every selected scalar crossing column, repeated/backward samples,
requested target IDs, original entry indices and input-file indices. Rows are
ordered by target argument order, then time, file index and entry. A requested
identity with no crossings remains in `target_id` with no associated rows; there
is no fallback by position or time. Existing output files are refused.

## Interpretation

Stock crossings are sparse pre-step samples, not continuous material histories;
no stock material column can be inferred from their momentum differences alone.
Fast x-plane intersections are chord interpolations, not validated VD hits.
Applied material order is sampler/dictionary order, not geometric within-step
order; episode positions bracket steps rather than exact material boundaries.
Selected histories cannot measure a population failure rate or establish
numerical convergence. Synthetic tests cover exact identities, repeats, empty
matches, chronology, callback mutation isolation and byte-identical transport
outputs with and without observation.
