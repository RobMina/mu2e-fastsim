# Field and thin-surface follow-up — 2026-10-27

## Scope

Opt-in synthetic diagnostics only. No application/physics defaults, C++ source, existing binaries or image
changed; no build or production transport. The Python h6 tools were copied from the existing unpublished
follow-up into this isolated tree; retained outputs are replayed rather than rerun. `boundary.py` and
`map_coverage.py` add bounded, independently checked diagnostics.

## Reused h6b control isolation

Existing A100 run 21983992 completed in 33 s, exit 0, with six clean memcheck processes; executable SHA256
`a423c659bda77c78b738fc50c5a63540caecf1807aa19f16beb7a395e10922a6` equals h6. Inputs and results retain their
original hashes. Each of six maps has 67 paths/backend. Eight smooth-field response gates pass, but the
**overall checker fails** eight paired records: the duplicated epsilon/cap point for p=5 MeV/c, q=-1,
L=600 mm, cap=16 mm, chord=20 mm, epsilon=1e-4 violates the unchanged 5e-4 momentum budget in constant and
affine maps at both resolutions. No threshold was relaxed or record excluded.

Constant coarse GPU, p=5 MeV/c, q=-1, L=600 mm:

| Varied control | Values | Endpoint errors (mm) | Work |
|---|---|---|---|
| epsilon, chord=20 mm, cap=16 mm | 1e-4 / 1e-6 / 1e-8 / 1e-10 | .254169 / 1.649e-4 / 2.404e-7 / 3.225e-11 | 784 / 1316 / 2632 / 6041 field evaluations; 38 outer steps each |
| chord, epsilon=1e-3, cap=16 mm | .5 / .02 / .005 / .0005 mm | .006898 / 5.323e-8 / 5.981e-11 / 7.128e-12 | 72 / 359 / 717 / 2266 outer steps |
| cap, chord=20 mm, epsilon=1e-4 | 16 / 8 / 4 / 1 mm | .254169 / .004771 / 2.840e-5 / 7.314e-10 | 38 / 75 / 150 / 600 outer steps |

The coarsest GPU momentum drift is 8.384e-4. Epsilon affects internal work despite unchanged outer counts.
The affine texture floor still precludes a claim of monotonic map/step convergence. Internal exhaustion is
not exposed. `delta_intersection` is not in the h6 request interface: it belongs to geometry propagation.

## New geometry-aware thin-surface diagnostic

A100 run **22043561**, 40 s, exit 0, node `udc-an37-19`. Existing h5 adapter SHA256
`6061c8f14695725f416e8e0e88a3fea5b8ea048afe84cb1455b651a50112c31c` matches retained h5 regression; loaded
Celeritas libraries resolve to and hash-match h5. No sanitizer was requested for this new job; h5's earlier
sanitizer evidence is separate.

The source is 10 MeV e-/e+, direction (0.6,0,0.8), in vacuum and Bz=1 T. A 10 micrometre thick lead surface
starts at z=20 mm. Two tiles share a boundary placed 1 micrometre to either side of the analytic helix
crossing. MSC is explicitly disabled for this diagnostic, not changed in any production default. Four events
per case; 32 GPU settings, four G4-only controls, four GPU zero-field controls. The field/charge changes move
the expected tile, so contained energy alone cannot satisfy the check.

**40/40 cases and 160/160 first deposited-hit tiles agree with the independent oracle.** All completed-event,
primary, hit, energy-sum and runtime CUDA-action checks pass. Predeclared arrival-time allowance is 1e-14 s.

| GPU setting | Changed values (lengths below converted from native cm) | Max first-hit time error (s) |
|---|---|---:|
| default | chord .25 mm, delta_intersection 1e-4 mm, epsilon .001 | 2.581e-17 |
| chord loose / fine | .5 / .005 mm | 1.676e-16 / 1.521e-20 |
| delta loose / fine | .01 / 1e-6 mm | 2.581e-17 / 2.581e-17 |
| epsilon loose / fine | 1e-4 / 1e-8 | 2.581e-17 / 1.893e-20 |
| joint fine | .005 mm / 1e-6 mm / 1e-8 | 1.521e-20 |

Delta-only first-hit identities/times match default for all four geometries, but full event JSONL differs in
two geometries: the material step energy changes while the first crossing proxy does not. Mean total edep
changes are -0.0884/-0.0888% (loose) and +0.0750/+0.0754% (fine), with no physics-closure inference from four
events per case. The insensitive first-hit observable is **not proof that delta is irrelevant**. The named epsilon_loose arm is tighter than the application default;
“loose/fine” denotes the two sweep endpoints. G4 uses different controller semantics (epsilon_step rather than
GPU epsilon_rel_max), so the controls are not asserted to have numerically identical integrators.

Limitations: score exposes first *deposit*, not a guaranteed first geometric crossing, position, length,
normal, navigation status or non-deposit steps. Material fluctuations/secondary histories remain active;
this tiny diagnostic is not a detector acceptance estimate or a precision-efficiency measurement. Surface
parallel/tangent trajectories, repeated crossings, nonuniform field boundaries and map-edge trial overshoots
still need an instrumented navigation probe, a user build, and a new predeclared validation suite.

## Mu2e readiness

Header audit of public Mau13 DSMap: 97x97x521 nodes at 25 mm, origin shift zero, x [-5096,-2696],
y [-1200,1200], z [3071,16071] mm; **no extendY**. It is already a full-y map. WorldMap has extendY flip=By;
only By reverses at negative y. Production composes DSMap/TSdMap/DSExtension ahead of WorldMap, then zero
where no map applies. The existing single-map adapter has no representation of this composite selection.

The checked Run1A GDML example world is 1063575.2 x 40732.8 x 1094531.2 mm. None of those map domains covers
it with the adapter's one-cell guard, even the reflected WorldMap. This example is not the matched DS reference
geometry. DSMap's 14,706,267 scalars fit the 30M cap; the cap is not the blocker. Payloads and release-matched
transforms remain unvalidated. The downloader checks object integrity, not the CVMFS root trust chain.

Adapter total edep cannot be relabeled StrawGasStep eIon. Future work requires nonionising subtraction or an
explicitly quantified equivalence, complete StrawId/ancestry, source-event ledger, times, lengths and stock
aggregation/cuts. The reference must inject the actual DS StepPoints, not confuse upstream GenParticles with
DS primaries. Plan the ladder Offline G4 -> adapter G4-only -> GPU with matched configuration, early-flash
prescale one, zero-hit events and event-cluster uncertainty. Do not use the TS3 Run1Baa runner unchanged.

## Replay and tests

From the workspace with the project Python environment:

```bash
E=remote-uva-data/mu2efastsim/probe
P=remote/analysis/fs_wt/flash/container
.venv/bin/python3 "$P/celeritas_adapter/field_probe/sensitivity.py" check \
  "$E/celer_field_h6b_21983992/results" --baseline "$E/celer_field_h6_21983494/results"
# Expected exit 1: the retained loose-setting momentum failures remain failures.
.venv/bin/python3 "$P/celeritas_adapter/boundary.py" check "$E/n5_boundary_22043561/results"
.venv/bin/python3 "$P/celeritas_adapter/map_coverage.py" "$E/n5_audit_20261027/"*.header \
  --gdml remote-data/crv_fastsim/geometry/mu2e_run1a_v01.gdml --mu2e-origin-world-mm 0 0 0
M2FS_FIELD_PROBE_RESULTS="$PWD/$E/celer_field_h6_21983494/results" \
  .venv/bin/python3 -m pytest -q "$P/tests"
```

Local suite: **70 passed, one optional pinned-upstream preparation skip**. Boundary replay is now read-only
and additionally rederives map/GDML plus verifies the retained h5 binary hash; the submitted runner snapshot
is unchanged. New runner deployment lives at `fastsim_uva/n5_boundary_20261027`; results at
`probe/n5_boundary_22043561`, audit/replays at `probe/n5_audit_20261027`. No rerun is needed for replay.
