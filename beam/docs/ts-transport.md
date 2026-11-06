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
The 12-run positive study also predates the enforced cap and is not convergence
evidence for this implementation. Held-out data remain untouched.

Use `python3 -m cProfile -o run.prof -m beamfastsim.tstransport ...` then
`python3 -m beamfastsim.tsbench --profile run.prof` for profiling. Small-batch
profiles locate hotspots; they are not full-run runtime predictions.

Run `test_tsfield.py`, `test_tsgeometry.py`, `test_tsphysics.py`,
`test_tstransport.py` and `test_tsbench.py` under `beam/tests/`. Set `TS1_GDML` and
`TS1_RUN1A_GDML` to saved matching geometries to include the two fixture tests;
the synthetic tests need no collaboration data. Local tests are not CI.
