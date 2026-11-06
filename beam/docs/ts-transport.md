# TS muon/pion transport and benchmark (opt-in)

Build an independent entrance cohort as described in [ts-reference.md](ts-reference.md).
The transport accepts one charged mu/pi lineage per entrance. Supply geometry and
field from the same stock configuration; never substitute Run1B DSOff for Run1A.
For Run1B Offline v13_39_00, the field-fetch default is DSOff:

```bash
python3 -m beamfastsim.tsfield field --release v13_39_00
python3 -m beamfastsim.tstransport --reference reference.npz --gdml geometry.gdml \
  --field-config field/offline/Offline/Mu2eG4/geom/bfgeom_DSOff.txt \
  --offline-root field/offline --maps field/maps --step 2.5 --seed 20261105 \
  --max-path 100000 --out fast.npz
python3 -m beamfastsim.tsbench --reference baseline.npz --candidate fast.npz \
  --out same-input-comparison.json
```

The CLI default step is 5 mm; pass 2.5 explicitly for the development operating
point. Angular cap is 0.1 rad; material chord and fractional mean-loss caps are
0.25 mm and 5%, checked on recomputed columns before sampling, including after
boundary truncation. Proportional shortening uses a 1e-6 relative margin; unresolved
lanes halve after five attempts. Failure to converge after 128 shortenings raises,
never accepts an over-cap step. Boundary retries recompute crossing flags without
consuming random draws. The 100 m cap is censored, not physical absorption.
Upstream exits use the bracketed entrance-plane intersection and its aperture,
not a fixed proximity threshold that can misclassify radial exits.
Select settings on development data, not held-out outcomes.

Field stepping uses core RK4/helix; field maps retain first-containing-map
precedence and declared Y reflection. Eight-corner interpolation preserves core
arithmetic order. Entry-field reuse and crossing-lane bisections reduce repeated
work while preserving exact DS-plane snapping. A conservative containment bound
can skip redundant quarter-step probes, never the full-step field/material checks.
For A=h*K*Bmax/p, the RK position bound is
h*max(1+A/2+A^2/4,1+A/2+A^2/6+A^3/24), with rounding padding. Unsupported geometry
or field safety methods use ordinary probing; `boundary_safety=False` enables
same-seed comparisons. Only changed material chords are recomputed.

Geometry consists of selected TS1–TS5 vacuum envelopes and material daughters,
with exact straight-chord intersections. It is not a general Geant4 navigator.
No pion nuclear interactions, wall return, stopped-pion daughters, polarization,
muon capture or decay electrons. Pion ionization is mass-substituted muon loss;
curved-path material ordering, time integration and grazing boundaries remain
finite-step approximations. Fixed map precedence differs from stock's cache.

After the review corrections, the same 25,078 Run1A development entrances take
187.215 s (133.953 entrances/s), one laptop CPU worker excluding field loading.
DS mu-minus: 1,088/1,098, ratio 0.9909 +/- 0.0219; mu-plus: 8/15, ratio
0.5333 +/- 0.1633. Paired-shard errors; sparse positive deficit remains. No matched
stock TS-only/GPU/SU speedup is measured. Settings and seed are unchanged, but
shortening changes RNG consumption; equal seeds do not make paired trajectories.
The preserved pre-review output (1,095/1,098 and 8/15, 132.820 entrances/s) and its
exact optimization comparisons are historical, not validation of corrected code.
The preserved pre-review positive study is not convergence evidence for this code.
A corrected rerun of the same 12 runs gives mean mu-plus counts 12.00 / 9.75 / 12.75
for baseline / tighter material / half field step (stock 15), with no path caps.
Four-seed spread and changing RNG consumption do not establish convergence or
justify choosing the closest yield. See the full TS guide for the separate tables.
Held-out data were untouched at that development checkpoint.

A subsequent all-charge joint tightening to 1.25 mm / 0.125 mm / 2.5%, same
25,078 inputs and seed, changes DS mu-minus 1,088 -> 1,073 and mu-plus 8 -> 9.
Mu-minus tighter/baseline is 0.98621 +/- 0.01985; setting-difference momentum/time
diagonal chi2 are 4.643/11 and 22.352/21. Zero live/path-cap/missing-field outcomes.
No clear population instability is resolved, but one stochastic joint change is
not convergence proof. This motivated retaining the baseline for the subsequently
authorized frozen held-out test. `tssensitivity` reuses the saved baseline and
never chooses settings by proximity to stock. See the [sensitivity guide](ts-sensitivity.md).

## Frozen held-out result (2026-11-06)

After the development checks, the unchanged baseline was run once on the disjoint
100-shard Run1A held-out reference: 500,000 counted Events, 25,334 known-root inputs.
DS mu-minus 1,164/1,153 gives fast/stock 1.00954 +/- 0.02371; mu-plus 9/11 gives
0.81818 +/- 0.16529, still sparse. Momentum/time diagonal chi2: mu-minus 14.283/11
and 12.128/21; mu-plus 4.015/4 and 10.030/11. Mu-minus elapsed-time overflow above
500 ns is 5 stock/1 fast, retained in yields, outside plotted-bin chi2. No fast
live/path-cap/missing-field outcomes. Stock DS pi-minus 7 versus fast 0 is a
separate limitation. Transport cost: 174.989 s, 144.775 entrances/s, one local
worker excluding field load; not an optimization speedup over the different
development sample. All held-out stock statuses are zero; 2,665 upstream ancestry
gaps and 5 nested entrances are explicitly retained.
This completes the opt-in comparison, not default promotion or convergence proof.
Held-out is now consumed; it must not become a tuning set advertised as a fresh test.

## Optional selected-lane history diagnostic

`transport(..., trace=observer)` copies selected source rows into `begin`,
`material` and `end` callbacks. The observer supplies distinct integer `rows`;
no RNG is supplied. `begin` records accepted field/boundary proposals, `material`
records actual sampler inputs/results, and `end` records updated states/fates.
Cold-at-start rows have terminal output but no accepted steps. Callback mutation
is regression-tested not to change physics arrays.

`python3 -m beamfastsim.tstrace --runs <existing-baseline-jsons> --out <fresh-dir>`
replays only development seeds 20261111/12/13, with the complete original positive
cohort. It saves two selected muons' histories only after every saved result and
exposure array matches (runtime excluded). `tscrossings` extracts their sparse
stock VD/DS records, and `tshistory --stock <npz> --traces <npzs> --out <json>`
compares recorded states and applied material episodes. Material call chronology
is not geometric within-chord ordering. Fast x-plane interpolation is not a
validated VD hit or a stock trajectory reconstruction. See the [history guide](ts-history.md)
for the bounded result; no physics controls or headline yields change.

Use `python3 -m cProfile -o run.prof -m beamfastsim.tstransport ...` then
`python3 -m beamfastsim.tsbench --profile run.prof` for profiling. Small-batch
profiles locate hotspots; they are not full-run runtime predictions.

Run `test_tsfield.py`, `test_tsgeometry.py`, `test_tsphysics.py`,
`test_tstransport.py` and `test_tsbench.py` under `beam/tests/`. Set `TS1_GDML` and
`TS1_RUN1A_GDML` to saved matching geometries to include all four fixture tests;
the synthetic tests need no collaboration data. Local tests are not CI.
