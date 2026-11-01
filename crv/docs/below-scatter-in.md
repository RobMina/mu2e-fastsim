# Experimental below-plane scatter-in (opt-in)

This is an algorithm-development mode, **not physics closure and not a new default**. The existing straight
transport and `below_msc=True` bar-to-bar algorithm are unchanged unless `below_scatter_in=True` is also passed.
No V1 gates, run defaults, GDML materials, or above-plane scattering formulae are changed here.

## Python API

Additional `crvfastsim.transport.transport` keywords:

| Keyword | Default | Meaning |
| --- | --- | --- |
| `below_scatter_in` | `False` | Enable the new marcher; requires `stage="C"`, non-None `below`, and `below_msc=True` |
| `scatter_in_step_mm` | `50.0` | Maximum straight segment, including segments inside bars; finite and positive |
| `scatter_in_envelope_mm` | `1500.0` | Finite near-miss admission/escape margin; finite and positive |
| `scatter_in_max_steps` | `20000` | Positive integer iteration cap; an unfinished track **raises**, never returns partial tables |
| `scatter_in_diagnostics` | `None` | Optional mutable dict receiving admission/termination/step counters |

The return remains `(steps, particles)`. `transport` and `runchain` expose `--below-scatter-in` and the three
`--scatter-in-*` controls; enabling also requires `--below-msc`. No RunConfig default is promoted.
As for existing below-MSC transport, pass no separate `absorbers`: `below` already contains the aluminium.
`below` must support the PieceSet interface (`mats`, `column`, `len`) and exclude scintillator bars.
Generated origins/momenta must be finite, with nonzero momentum. Cuts must be positive, finite and one per bar.
New controls are validated only when this mode is enabled; disabled calls keep historical behavior/RNG draws.

```python
diag = {}
steps, particles = transport(
    gen, geom, rng, stage="C", world=above_world, below=below_world,
    below_msc=True, below_scatter_in=True,
    scatter_in_step_mm=50., scatter_in_envelope_mm=1500.,
    scatter_in_max_steps=20000, scatter_in_diagnostics=diag,
)
# Save diag alongside producer inputs, source hashes, seed and output tables.
assert diag["complete"] and not diag["cap_hit"]
```

The lower-level `scatter_in_crossings(..., *, step_mm=50., envelope_mm=1500., max_steps=20000,
diagnostics=None)` accepts plane states and returns twelve arrays: track, bar, first entry position/direction,
first entry momentum, summed bar path, summed raw bar loss, first entry distance, last exit position/momentum,
last exit distance, and summed segment-visible energy. Track indices refer to its input arrays, not event IDs.

## Admission and finite domain

* Before above-plane transport, the straight downward ray must reach the CRV bounding box expanded by
  `max(1500 mm, envelope_mm)`. Thus a wider below-plane admission never bypasses above-plane loss/MSC.
* After above-plane transport, surviving processed tracks whose downward ray reaches the box expanded by
  `envelope_mm` are admitted, **even if they miss the exact box or every individual bar**.
* Each admitted track marches from its plane point. Its finite domain encloses both the detector bounding box
  and its own initial point, expanded by `envelope_mm`; the upper y face is the plane (or initial y if higher).
  This includes the long below-plane approach of a near-horizontal side-wall track. Per-track domains do not
  depend on other tracks in a batch. Leaving any domain face is absorbing; an endpoint kick cannot recover it.
* This is a finite acceptance approximation, not an assertion that tracks outside it cannot scatter in. Upward
  initial rays and rays rejected before above-plane processing are not recovered. Increasing the envelope must
  be studied independently of step refinement.
* `below_margin` in `load_below` controls **material availability, not admission**. A large marching envelope or
  long near-horizontal approach is not useful if its material was omitted at loading. Increase/check material
  coverage separately; the marcher cannot infer absent GDML pieces. Unknown materials retain the existing
  concrete fallback, and PieceSet overlap/surface approximations remain.

## Marching and ledger

A vectorized active-track loop chooses the minimum of the configured step, next bar entry/exit, and domain exit.
It integrates only that segment's passive material or scintillator, advances through the segment, then applies
an angular Highland kick at its endpoint. All bar intersections are recomputed after every segment/kick.
Having no next bar is **not a termination condition**. No future candidate-bar gap is prepaid, and no already
traversed segment is intentionally charged again when a kick changes the destination. Passive solids replace
air in this mode; air has mean loss but no below-plane scattering, as in the legacy below-MSC approximation.

Scattering uses cumulative x/X0 including the above-plane contribution and the core Highland variance increment
at geometric-mean segment momentum. There is **no lateral endpoint displacement**: it could jump across a thin
bar without recording a crossing. This first-order, endpoint-angular diffusion underestimates lateral spread
for coarse segments and approaches distributed scattering only under refinement. It does not preserve the
legacy single-slab angle/displacement covariance at finite step. Passive material within a segment is charged
in fixed material-name order, not resolved surface order; make the step small enough to study this error.
The standard small-angle Gaussian model, including its limitations near range-out/large angles, remains.

Inside bars, segments are boundary-clipped and may turn before exiting. Energy is capped at residual kinetic
energy. A stopping segment's path is estimated by the residual-energy fraction of its sampled full loss
(uniform-loss approximation), and the track never restarts. A passive stop creates no later bar. Non-stopped
tracks below the existing 1 MeV/c tracking threshold also terminate.

One row per `(run, subrun, event, sp, bar)` is retained. The first entry position/direction/momentum and last exit
position/momentum are saved; paths, raw loss and segment Birks-visible energies are summed. A physical re-entry
adds its genuinely travelled bar path/energy to that ledger, not another truth-identity row. Consequently the
endpoint chord need not equal the summed bar path; gaps between visits are not bar energy or length. `p1` is the
actual final exit momentum, not first-entry momentum minus bar loss alone (there may have been passive losses
between visits). Particle keys and the full primary-particle table are preserved. Storage grows with recorded
bar segments before reduction; there are no per-track Python loops, but the active segment loop can be costly.

## Diagnostics and convergence requirements

Diagnostics include `generated_muons`, `above_processed`, `admitted`, `tracks_started`, `escaped`, `stopped`,
`segments`, `bar_segments`, `iterations`, `max_segments_per_track`, `max_path_mm`, `max_segment_mm`,
`max_segment_xx0`, `max_scatter_sigma_rad`, `recovered_initial_misses`, `cap_hit`, `complete`, and the three control values. Recovered misses counts tracks
whose **initial post-plane ray missed every bar** and later traversed one; it is not an efficiency estimator.
The counters describe algorithm completion, **not convergence**. A cap exception leaves diagnostics available,
but tables must not be persisted from a partial run. Unexpected invalid sampled losses also raise.

Before interpretation or promotion:

1. Run fixed generator inputs with step, step/2 and step/4. Compare all-sector crossing rates, matched entry
   momenta, bar paths/energies, recovered/missed populations, stopping and path strata, with event-clustered
   uncertainties. The toy covariance-refinement test is necessary algorithm evidence, not a physics tolerance.
2. Independently increase the admission/escape envelope and the loaded material margin; test near-horizontal
   side-wall tracks and ensure admitted approach paths have material coverage. No finite default is certified.
3. Require no caps; record the worst segment count and runtime/memory before scaling. Raising the cap only fixes
   truncation, not convergence. Changing controls changes RNG draw order; same seed is not trackwise correlated
   scattering and bitwise equality is not expected for the enabled mode. Even zero scattering with subdivided
   stochastic losses need not be seed-identical to legacy full-chord sampling.
4. Apply the full persisted-table validation/V1 workflow in `transport_validation.md` (branch `park/named-run2`), retain all sectors and
   producer hashes, and inspect matched/unmatched truth identities. No V1 result is provided by these tests.

**Timing is still approximate physics.** Table times use accumulated polygonal flight distance divided by the
*generated* beta, not a segment-by-segment slowing-muon time integral. Last-exit time includes gaps between
re-entries. The above-plane offset retains its existing approximation. There is no below-plane tail model,
continuous curved-path correction, decay model, or timing-closure claim here.

## Local algorithm evidence

Tests: `tests/test_transport_scatter_in.py` covers deterministic no-initial-hit recovery, miss-after-kick recovery,
re-entry ledgering, stops, bounded material queries, deterministic zero-scatter equivalence, multiple-muon
identities, cap/invalid inputs, unchanged disabled output/RNG, coordinated above-plane admission, finite-envelope
rejection, and constant-power slab endpoint-covariance refinement. Existing below-MSC, energy-ledger and tail
tests also run unchanged. The test command used no bytecode/cache writes:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CORE:$PWD" \
  "$PYTHON" -B -m pytest -p no:cacheprovider \
  tests/test_transport_scatter_in.py tests/test_transport_below_msc.py \
  tests/test_transport_energy_ledger.py tests/test_transport_msc_tail.py -q
```

Initial algorithm tests used core `a554571`; integration and real-shard probes use a read-only export of
`a35760693cf68d65d2a8b18f0a1a080b12cacd03`, not the stale main worktree.
A synthetic real-loss/real-scattering smoke of 1,000 tracks, three 10 mm-thick wide bars, empty passive world,
100 mm envelope gave 3,000 rows at steps 50/25/12.5 mm, taking 0.150/0.047/0.049 s locally (first-run overhead;
9,000/11,000/15,000 segments). All steps exceed the bar thickness: **this is not a scattering convergence
measurement**, a named-geometry/per-file timing, or a basis for campaign scaling. Real GDML/module lookups and
long near-horizontal paths may be much more expensive; measure one production file first.
