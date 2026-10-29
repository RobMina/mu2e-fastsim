# Bounded keyed SiPM diagnostic

`veto.bounded_sipm` is a separate diagnostic adapter with contract
`diagnostic-sipm-bounded-v1`. It does not modify production response, the frozen
`replay_sipm_control` helper, the qualified-response adapter, or saved outputs.
It is not stock sequential-queue equivalence or campaign admission.

## API and per-call limits

```python
from crvfastsim.veto import bounded_sipm as B

limits = B.Limits()
optical = B.optical_candidates(photons, photon_map, cfg, seed,
                               pixel_policy="fixed_integer", limits=limits)
dark = B.thermal_candidates(live_channels, cfg, seed,
                            start=100.0, end=1500.0, limits=limits)
charges, attempts = B.simulate_candidates(candidates, cfg, seed,
                                         end=1500.0, limits=limits)
```

`candidates` is the caller's combined optical/thermal candidate table. Return
schemas and ordering match the frozen helper: candidate columns are
`event, chan, root, time, px, py, dark`; both simulation products have the complete
frozen attempt schema, including ancestry, generation, multiplicity, uniforms,
probabilities, inactive/fired flags, charge and PE. Charges are fired attempts.
No admission receipt or partial product is returned on failure.

`Limits` is an immutable frozen dataclass, validated at construction and again
at every call. Fields are nonnegative built-in Python integers; booleans, NumPy
integer scalars, floats, negative values and out-of-domain integers are rejected.
Zero is permitted and means no capacity for that resource. Count limits have a
hard ceiling of 10,000,000; generations cannot exceed 1682. `state_bytes` cannot
exceed the platform signed indexing maximum, and the live-channel ceiling also
respects the platform state-indexing domain. Products used for admission are
Python integer arithmetic, not overflowing NumPy integer multiplication.

| Limit | Default | Admission meaning |
|---|---:|---|
| `photons` | 300,000 | Optical input rows; also optical candidate rows at simulation entry |
| `thermal_candidates` | 300,000 | Thermal output rows and dark candidate rows at simulation entry |
| `candidates` | 1,000,000 | Optical output/combined simulation input rows, including late input rows |
| `attempts` | 1,000,000 | Aggregate simulation ledger rows and conservative neighbour expansion allowance |
| `generations` | 1682 | Number of admitted generations per root, counting primary generation zero |
| `state_bytes` | 314,572,800 (300 MiB) | Dense float64 recovery state allocation |
| `live_channels` | 20,480 (4 × 5120) | Event/channel pairs, not distinct detector-channel labels alone |

Defaults accommodate at most four fully live 5120-channel frames by the
live-channel budget; the adapter does not separately limit the number of sparse
frames. The all-active default state is 275,415,040 bytes, below 300 MiB. Caller
limits may be smaller or larger within the validated domains. They are explicit
per-call limits, not a process-wide or cumulative campaign budget.

## Guards before named allocations

* Optical row budgets are checked before copying/validating the photon frame or
  delegating pixel mapping. Unique event/channel pairs are admitted before the
  frozen keyed pixel mapper runs. The mapper itself is unchanged.
* Thermal input channel rows are admitted before copying the channel table and
  allocating the wait-loop arrays. Duplicate pairs remain invalid. The expected
  count must fit the minimum of thermal, candidate and attempt limits, but this
  is only an early conservative guard, **not** a bound on a Poisson draw. Every
  vectorized wait round checks the actual accumulated count before copying its
  slot rows and allocating its negative-root array. Concatenation therefore only
  sees already-admitted rows. Negative roots remain `-1 - ordinal`.
* Simulation checks combined input count before copying it; optical/dark budgets
  apply before the unchanged inclusive end cut. After that cut, active
  event/channel pairs must fit `live_channels`, and
  `active_pairs × 41 × 41 × 8` must fit `state_bytes` before `np.full` allocates
  recovery state. The padded 41 × 41 state still supports legacy pixel 40.
* Every rank/cascade round checks aggregate ledger count and generation count
  before allocating its attempt-row arrays or appending the DataFrame.
* Before allocating the four-neighbour matrices, the current aggregate count
  plus `4 × fired_frontier_count` must fit `attempts`, and the next generation
  must fit `generations`. This deliberately overcounts boundary losses,
  nonproduced neighbours, and merged duplicate hits. It can reject a realization
  whose final ledger would have fitted; rejection never clips the expansion.

Resource failures raise `RuntimeError` with the named limit and an incomplete
batch message. Invalid limits, inputs/configuration or nonfinite result
arithmetic raise `ValueError`. No limit changes probabilities, renormalizes an
exposure, drops excess photons, or returns a successful partial ledger. The
ordinary frozen end-window cut is unchanged: end is inclusive, with no lower
candidate time cut. Nonfinite thermal windows, thermal means, candidate times,
and avalanche arithmetic are rejected, not returned as valid output.

## Physics, identities and whole-frame recovery

The minimal thermal-wait and rank/cascade algorithms are copied only to insert
admission checks. Configuration/frame validation, empty schemas and keyed draws
reuse `replay_sipm_control`; the neighbour geometry is the same helper. No global
limits or production functions are monkeypatched. The same hash and
root-relative pixel contract, zero-trap/40 × 40 restrictions, breadth-first
multiplicity law, inactive pixels, thermal law, tie ordering and recovery
arithmetic apply. Both pixel policies are supported; `fixed_integer` is the
intended diagnostic path. Loops are over thermal wait rounds, root ranks,
cascade generations and fixed columns, never events.

Each call processes the entire supplied recovery history without internal
splitting. A frame too large for its limits fails. **The caller must supply all
roots for each frame together**: these table-only functions cannot detect a
caller splitting one frame across separate invocations. Do not retry an
oversized frame as root subsets or time slices. Splitting unrelated whole frames
is an executor/planner responsibility.

Thermal input must include all status-live event/channel pairs, including
optically quiet ones. Empty outputs cannot encode the frame denominator.
Keep the complete qualified frame registry externally and retain every frame
through empty optical, thermal, charge and attempt outputs. These functions do
not build, filter or replace that registry, and do not restore full input-SHA/run/
subrun/event identities on their own. Registry-derived seeds and labels must
come from the calling diagnostic executor.

## Evidence and limitations

The single-file synthetic suite checks exact values, dtypes, columns and ordering
against the frozen helper for optical mapping, keyed thermal counts and roots,
combined optical/thermal response, same-pixel recovery, inactive pixels,
multiplicity and a full saturated cross-talk cascade. It also checks unrelated
subset/permutation behavior, finite numeric outputs, empty and late schemas,
external zero-output frame retention, invalid limits/configuration/identities,
unchanged global RNG state and fail-before-allocation spies. Spies replace only
adapter-local references; neither the frozen helper nor production is patched.

Run only this suite from the repository with the required environment/dependency
path configured:

```bash
python3 -m pytest -q tests/test_veto_bounded_sipm.py
```

This is within-environment synthetic exactness, not a cross-platform floating
point guarantee or validation on retained real assets. Caps bound the named
arrays/counts, not peak RSS, simultaneous temporaries, pandas sorting/storage,
input photon creation, ROOT decompression, output bytes or disk quota. Optical
input already exists at call entry; an upstream photon generator needs its own
pre-allocation guard. A concatenated input table likewise already exists before
simulation admission. Conservative upper bounds may reject otherwise small
outputs. Executor integration, artifact transactions, full provenance binding
and atomic completion receipts remain separate work; any exception must leave
the entire batch/run incomplete rather than silently remove exposure.
