# Flash execution-mode regression contracts

The opt-in schedulers compile different JAX graphs. Material lane order, shape,
RNG iteration, event decisions, energy and record ownership are still required
to agree exactly in the regression fixtures. Byte identity of every floating
leaf across different compiled graphs is not a portable CPU contract.

## Reproduced hosted CPU differences

With Python 3.13, JAX/jaxlib 0.11.2 and NumPy 2.5.4 on Ubuntu 24.04,
independent AMD EPYC 9V45 and 9V74 runners reproduced the same four failures;
AMD EPYC 7763 passed the original assertions. Full raw-state diagnostics found:

- Compact-tail versus full-graph fixture: one direction component differs by
  two float64 ULPs (2.168404344971009e-19). All other leaves agree byte-for-byte.
- Tilted-field positron fixture, `skip_empty=True`: one accumulated time differs
  by one ULP (3.469446951953614e-18 ns). Other differences are confined to the
  already documented continuous-loss, path and coordinate/direction budgets.
- Full versus split execution of each individual compact graph remains exact.
- An additional diagnostic comparison of skip-only versus full execution of the
  compact-tail fixture differs by three ULPs in direction. This is not the
  compact-versus-full test and is not covered by its two-ULP allowance.

These are measured graph/backend-dependent rounding differences, not a proof
of a particular compiler optimization or universal parity for future geometries.
No transport expression or random scheduling is changed to accommodate them.

## Test allowances and protections

`test_flash_vacuum_bucket.py` uses a two-ULP bound only for the final direction
of the compact-versus-full fixture. Every other leaf, outer loop counter and
same-graph full/split comparison stays byte-exact. Initially inactive position,
time and path sentinels are checked against the initial state. The separate
long-vacuum fixture retains its existing 1e-10 coordinate/direction budget.

`test_flash_vacuum_fast.py` permits one ULP only for accumulated `t` in the
existing tilted-field cross-mode comparator. Its other named absolute budgets
are unchanged. Record validity and invalid record payloads are byte-exact.
The ULP comparisons require finite float64 arrays with identical shape and
dtype, and preserve exact zero bytes (including signed zero). Neither allowance
applies to same-program chunking, energies, fates, RNG clocks, masks or counts.

`test_flash_mode_comparison.py` mutation tests accept the specified limits and
reject the next ULP, nonfinite values, zero/subnormal changes, dtype/shape/schema
changes, and changes to the protected state. A new backend discrepancy must be
diagnosed by field; these limits are not a general permission to relax tests.
