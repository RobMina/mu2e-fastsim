# Thread B: waveform and conditions regression contract

## Scope (2026-10-25)

First working-tree repair against CRV `837f5ea`, core `a554571`. No commit, build, remote compute job, stock run,
GPU run or new physics result accompanies this change. Concurrent transport and objective edits are separate work.
The frozen A0/A0'/A0g outputs are unchanged.

The general chain and the DQM table exporter must not lose pulses because an input exceeds a numerical capacity,
forget per-channel timing conditions, or retain a channel marked ignored for pulse reconstruction.

## Implementation

- `reco/chain_jax.py` carries the original-input truncation flag alongside the device output. Truncated inputs,
  exhausted pulse rounds and overflowed fit windows use full-length host recovery. This applies even when the
  first twelve samples contain no peak. Output waveform indices and sequence order are retained.
- `reco/pulsefit.py::reconstruct_full` uses the valid sample count as a safe upper bound on pulse rounds: each
  successful peak search advances the fit-window end. The vectorised reconstruction stops when no peaks remain.
  The low-level bounded `reconstruct` interface remains available for kernel comparisons. Full recovery is used
  by the general NumPy chain, JAX fallback, validation CLI fallback and `valhist.fasttables.fast_reco`.
- `chain.reco_waveforms` carries DB offsets. `chain.clusters` supplies them when a caller's waveform dictionary
  lacks them, without overriding an explicit offset/time-calibration array. Each path adds the offset once.
- Pulse reconstruction removes status bit 1 (mask `0b010`) before coincidence finding and member-index creation.
  Status bits 0/2 are not pulse-reco vetoes: their charge-generation suppression belongs upstream. The unchanged
  full status array still reaches the coincidence finder for partner compensation.
- The DQM table exporter retains original waveform indices when filtering pulses, so exported digi links remain
  valid; peak and leading-edge times both receive the same DB offset.
- Empty host reco returns typed empty indices/flags; the JAX chain accepts an empty waveform population.
  Ignored/quiet waveform events are retained in the JAX event census.

## Executed verification

Laptop `~/mu2e/.venv/bin/python3`, JAX 0.11.2 CPU; no new dependencies. The initial new regression file gave
**13 failures / 1 pass** before the repair. The expanded final contract has **24 tests**.

```bash
cd ~/mu2e/remote/analysis/mu2e-fastsim
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core:crv ~/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
  crv/tests/test_chain_reco_contract.py crv/tests/test_reco_pulsefit.py crv/tests/test_chain.py \
  crv/tests/test_reco_jax.py crv/tests/test_background.py crv/tests/test_runconfig.py \
  crv/tests/test_objective.py crv/tests/test_timecal.py
```

**69 passed in 43.71 s**, with `git diff --check` clean. This is the scoped regression suite, not all CRV tests.
Coverage includes late-only and separated peaks, a peak at the input boundary, ten pulses, quiet long inputs,
mixed long/short rows, nonidentity row order and padded batches, failed-fit flags, heterogeneous offsets with
and without explicit calibration overrides, all eight status-bit combinations, digi links, and empty inputs.
A real NumPy/JAX coincidence comparison with ignored channels and unequal offsets agrees in member identities
for one and three chunks. Other glue tests mock geometry/coincidence calls to isolate the boundary contracts.
The NumPy port is the oracle here; these tests are not a new comparison against Offline's pulse fitter.

## Still open before promotion

1. Residual fused-response overflow must fail or recover before output/scoring; unsupported thermal/trap settings
   and windows beyond the fused photon support need an explicit backend contract. These are not fixed here.
2. Empirical background pulses are inserted after pulse reconstruction; their channel-status policy and treatment
   under changed conditions require separate qualification. This patch filters waveform-derived pulses only.
3. Measure long-waveform/fallback prevalence and runtime on real inputs. No speed claim follows from these tests.
4. Exercise the same contract on supported GPU backends and stock reference digis, including round-trip products.
5. Capture immutable source/conditions/library/input provenance before uniquely tagged post-aluminium A1/E2E
   candidate runs. Concurrent transport corrections must not be described as an aluminium-only intervention.
6. Add histogram inventory/dimension/flow coverage and uncertainty calibration. Separate rate from shape in
   PE/X/Y/NPc diagnostics; current TH1 heuristic flags cannot certify full DQM acceptance.
