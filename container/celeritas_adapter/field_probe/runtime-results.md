# h6 field/ODE runtime results — 2026-10-26

## Verdict and limits

**Synthetic field-value, endpoint-budget and map-query-safety checks pass on host and A100**, using the
unchanged user-built h6 executable and six original fixture cases. All six Compute Sanitizer processes
report zero errors. The original Python checker failed an exact step-count assertion; two checker-only
corrections were required for replay, documented below. No physics accuracy thresholds were changed.

This is **not** geometry-aware propagation, material/MSC transport, real Mu2e map coverage, ionising-edep
scoring, DS-start-reference agreement or speed/ranking validation. Adaptive-controller convergence is not
certified: several epsilon sweeps plateau, and affine texture errors need not improve monotonically with
step size or map refinement. The diagnostic error envelopes are not Mu2e acceptance tolerances.

## Build, execution and identity

| Item | Evidence |
|---|---|
| User build | Slurm **21983481**, COMPLETED, exit 0:0, 07:25:11–07:25:27 EDT (16 s), `udc-an34-31` |
| Runtime | Slurm **21983494**, COMPLETED, exit 0:0, 07:28:47–07:29:01 EDT (14 s), `udc-an37-25` |
| Device | NVIDIA A100-SXM4-80GB, driver 595.71.05 |
| Source manifest | `7b6c29280c469ac98a21c5ec26e352c96c78734f115b16e36b70c28f133c78bf` |
| Executable SHA256 | `a423c659bda77c78b738fc50c5a63540caecf1807aa19f16beb7a395e10922a6` |
| Submitted wrapper SHA256 | `d685fd772483471f42f450c497cb027f7ae4768bdf52476f8e3806e276af0a03` |
| Existing library prefix | `/scratch/ram2aq/mu2efastsim/celer_rdc_h5/install` |

The five build steps include `ProbeKernels.cu` compiled with **`-rdc=true`**, a device link against h5 static
archives and image VecGeom, and the executable link. Cache/compilation database bind the pinned source,
map parser and h5 installation; binary timestamp is newer than source. Runner checks every source hash,
the pinned h5 library hashes and fresh `ldd`: loaded `libceleritas`, `libgeocel`, `libcorecel` all resolve to
h5, with no missing dependencies. H6 owns its final device link; no separate `libaccel_final` is required
in this executable's loaded family. Geant4/VecGeom remain image dependencies.

Evidence root on UVA:
`/scratch/ram2aq/mu2efastsim/probe/celer_field_h6_21983494/`, plus `celer-h6-21983494.out`.
Mirrored under `~/mu2e/remote-uva-data/mu2efastsim/probe/`.
The root holds build logs/cache, compiler database, source manifest, Slurm accounting and hashes;
`results/` holds the unchanged maps/requests/expectations, raw host/device outputs, memcheck logs and run
manifest. `oracle-check-v3.json` is the successful corrected replay. `replay-source/` and
`replay-provenance.sha256` retain the checker actually used. Raw result/fixture hashes were verified after
sync; the checker independently regenerates all six fixtures and reference solutions.

No h4/h5 baseline was rerun, no agent build was started, and no Mu2e-internal input was transferred.

## Measured values

Each map has **1,016 queries** (968 accepted, 48 rejected across all six inset faces) and **96 trajectories**
(90 completed paths, six intentional first-trial guard crossings), on **each** backend. Across six maps:
6,096 query records and 576 trajectory records per backend. Completion tags, paths, charges, momentum
conservation, field-evaluation accounting and unchanged retained states after guard rejection pass.
There are no numerical-failure records. The guard rejects before interpolation; clamping is not used as
an outside-domain policy.

| Map | Max host field component error (T) | Max GPU field component error (T) | Max host endpoint error (mm) | Max GPU endpoint error (mm) |
|---|---:|---:|---:|---:|
| zero coarse/refined | 0 | 0 | 6.41e-11 | 6.41e-11 |
| constant coarse | 1.95e-7 | 0 | 1.67e-4 | 1.65e-4 |
| constant refined | 1.95e-7 | 0 | 1.64e-4 | 1.65e-4 |
| affine coarse | 1.88e-7 | 8.12e-5 | 1.64e-4 | 0.0171 |
| affine refined | 2.20e-7 | 4.42e-5 | 1.67e-4 | 0.0312 |

Field entries are maxima over queried points and components; trajectory entries are maxima over all
accepted fixed/adaptive variants, not a matched-track refinement ratio. All component errors stay within
the **predeclared** float/texture bounds. Largest final relative momentum-magnitude drift is 5.76e-7.
The refined affine worst endpoint is `p5_q-1_L600_adaptive_16_1e-07` (0.0311806 mm); its conservative field
endpoint envelope is 1.05352 mm. Passing this envelope does not supply a precision requirement for Mu2e.

### Sweep interpretation

- Constant GPU, 5 MeV/c, q=-1, 600 mm: fixed caps **4 / 1 / 0.25 mm** yield endpoint errors
  **2.840e-5 / 7.314e-10 / 2.643e-11 mm**, demonstrating field-sensitive fixed-step convergence for this case.
- Same constant trajectory, adaptive epsilon **1e-3 / 1e-5 / 1e-7**: all take 227 substeps, with the same
  **1.483e-6 mm** error. This is not evidence of epsilon-controlled convergence. The chord constraint may
  dominate; internal substepper iteration-exhaustion status is not exposed.
- Affine coarse GPU, same trajectory, fixed caps: **0.01478 / 0.01707 / 0.01690 mm** errors. The nonmonotone
  behavior is consistent with the predeclared texture-error floor, not smooth double interpolation.
- Refining the affine grid reduces the worst sampled field component error **81.2 → 44.2 microtesla**, but
  the worst trajectory endpoint error **increases 0.0171 → 0.0312 mm**. Do not claim uniform trajectory
  improvement or a general convergence order from this suite.

## Checker corrections, not a transport change

1. The original checker demanded `ceil(600 / 0.25) = 2400` fixed steps. Native CGS double accumulation
   reaches `59.99999999999766 cm` after 2400 steps; the remaining **2.34e-11 mm** exceeds the probe's
   **4.26e-12 mm** termination allowance. A tiny 2401st step is therefore legitimate, with **16807**
   field evaluations. `fixed_step_count` now replays only this scalar path bookkeeping and still requires
   an exact step/evaluation count. The independent helix/DOP853 references and all field, endpoint, path,
   momentum and boundary thresholds are unchanged. Missing/extra counts remain negative regression tests.
2. Successful report serialization exposed a `numpy.bool_` in `possible_field_floor`; it is now explicitly
   converted to Python `bool`. This changes no acceptance decision.

Original failure is reproducible with the immutable snapshot and retained in
`original-oracle.{stdout,stderr,exitcode}.txt`. The empty `oracle-check.json` and `oracle-check-v2.json` are
failed replay attempts, not accepted reports. No raw output, immutable source snapshot or built binary was
edited to obtain the corrected pass.

Local suite with saved-result replay: **84 passed, one skipped**. The skip is the optional upstream-source
preparation test (`M2FS_CELERITAS_SOURCE` not set), not a runtime failure. Added regressions cover native-unit
step bookkeeping, rejected missing/extra counts, strict JSON serialization of real results, and an absolute
helix-radius check derived independently from SI charge, momentum and speed-of-light units.

## Reproduce without rerunning GPU transport

```bash
cd ~/mu2e
E=remote-uva-data/mu2efastsim/probe/celer_field_h6_21983494
.venv/bin/python3 "$E/replay-source/oracle.py" check "$E/results" > /tmp/h6-replay.json
M2FS_FIELD_PROBE_RESULTS="$PWD/$E/results" .venv/bin/python3 -m pytest -q \
  remote/analysis/mu2e-fastsim/container/tests
```

The executed wrapper is `field_probe/run_probe.sh`, deployed outside immutable `src/` at
`/home/ram2aq/mu2e/fastsim_uva/celer_field_h6/run_probe.sh`; it requests one A100 and runs only the existing
binary under memcheck. No new run is needed to reproduce the local verdict.

## Next gates

1. Separate chord and epsilon sensitivity, and define stronger smooth-field convergence tests before
   claiming adaptive-controller convergence; add representable/nextafter map-face regressions for other grids.
2. Field-sensitive **geometry-aware** propagation through known surfaces: crossings/endpoints, step/boundary
   convergence, and host/GPU controls using the repaired Cartesian action rather than only this ODE harness.
3. Only then tackle real Mu2e map transforms and whole-world coverage, explicit outside-map policy, total
   versus ionising energy scoring and event/primary denominators, followed by the matched DS-start reference.

No rebuild is needed for replay. A future C++ probe/adapter change must be handed to the user to build.
