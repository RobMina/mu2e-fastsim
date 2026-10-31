# beam/: Geant4-free muon beam transport to the stopping target (prototype)

Package `beamfastsim` (imports `mu2efastsim` from `core/`). Replaces the DS stage of the stock beam chain
(`Production/JobConfig/pileup/MuBeamResampler.fcl`) for the target-stop question: muons from the DS entrance (as
written by the POT stage, `MuBeamCat`) are carried through the field map, the stopping-target foils and the IPA in
batched JAX lanes (CPU or GPU) to their stop, decay or exit.

No Mu2e-internal data or results are stored here; the field maps, geometry and reference samples are read at FNAL.

| Path | What |
|---|---|
| `beamfastsim/fieldmap.py` | G4BL `.header/.bin` maps; trilinear lookup as Offline `BFGridMap` |
| `beamfastsim/simpleconfig.py`, `geometry.py` | Offline geometry text files -> foils, TSdA, IPA, outer PA |
| `beamfastsim/transport.py` | RK4 stepping, layer energy loss (G4 Urban via `mu2efastsim.eloss_jax`) and Highland scattering, range-out, decay, lane compaction |
| `beamfastsim/g4ref.py`, `compare.py` | per-muon G4 reference table and fast-vs-G4 metrics/plots |
| `beamfastsim/bench.py` | throughput on synthetic muons (no inputs needed) |
| `fcl/g4ref_mubeam_ntpart.fcl`, `art/run_g4ref.sh` | stock DS stage of SimJob MDC2025ac with a per-muon tree |
| `art/run_fast_cpu.sh` | g4table + field check + fast run + comparison on a GPVM core |
| `art/eaf_beam.sh` | the same plus a benchmark on an EAF A100 slice (`core/eaf/eafhub.py run`) |

```bash
# GPVM: G4 reference (stock DS stage, ~12 ms/muon), then the fast arm on the same muons
bash beam/art/run_g4ref.sh $M2FS_DATA/beam_fastsim/g4ref_full 214310 1
bash beam/art/run_fast_cpu.sh $M2FS_DATA/beam_fastsim/g4ref_full $M2FS_DATA/beam_fastsim/fast_v1_20k 20000
# laptop/any: unit tests and benchmark
PYTHONPATH=core:beam python3 -m pytest -q beam/tests
PYTHONPATH=core:beam python3 -m beamfastsim.bench --n 2000
```

## Persisted reference audit

Audit existing outputs before reusing a partial reference; this does not run transport or require JAX/CVMFS:

```bash
PYTHONPATH=core:beam python3 -m beamfastsim reference-audit \
  --nts "$REF/nts.owner.beamfs_g4ref.version.sequencer.root" --g4 "$REF/g4ref.npz" \
  --artifact "$REF/job.fcl" --artifact "$REF/job_config.txt" \
  --artifact "$REF/job.log" --artifact "$REF/mubeamcat.txt" \
  --out "$REF/reference_audit_UNIQUE_TAG.json"
```

The audit selects the highest `mutree/ntpart` cycle, records every available cycle's entry count, counts distinct
persisted event/primary IDs, hashes the ROOT/table/configuration artifacts, and compares all raw primary columns to
the cached table in row order. Duplicate primary IDs or a table mismatch make the CLI exit nonzero.
Add `--geom PATH --offline-root ROOT` to recompute `q`, `at_rest` and `foil` from ROOT rows, compare them to the
cache, and record the effective geometry plus SHA-256/size/path of every file traversed by the beam geometry reader.
`--baseline-geom Offline/Mu2eG4/geom/geom_common.txt` additionally checks the parsed overlay contract below.
These are current-reader checks, not historical G4 binary/geometry certification. Empty-muon events and source art
contents are not certified. Review the saved job outcome separately; particle rows are not event counts.

`transport_all` enforces an exact positive-integer step cap, including the final partial chunk. Foil crossings own
the arrival endpoint and exclude departure in either direction. `make_stepper` rejects nonfinite/malformed foil
arrays, invalid radii/thicknesses, overlapping or unordered foils, and centre spacing smaller than `ds`. Treat the
geometry arrays as immutable after compilation. This is a stepping precondition, not a full geometry contract:
the inherited material, support and absorber approximations are not certified by this check.

`run` and therefore `point` reject effective parsed changes outside these target-overlay keys, relative to
`Offline/Mu2eG4/geom/geom_common.txt` in the selected Offline root:

- `stoppingTarget.{radii,halfThicknesses,zVars,deltaZ,z0InMu2e,holeRadius,holeRadii}`
- `protonabsorber.distFromTargetEnd`

This includes changes through nested includes, not just the top-level file. A scalar hole change shadowed by a
per-foil `holeRadii` vector is rejected. The baseline's unmodelled supports/absorbers remain approximations; parsed
material, tilt, field-axis and unrelated detector changes are rejected rather than silently scored. This guard
covers effective **parsed** keys only: the minimal SimpleConfig reader is not a general syntax validator, and a
user-selected Offline root is not an immutable approved baseline. Full G4 support for per-foil holes is release
dependent. Allowed target changes still need paired physics validation; they are not a certified BO search space.

For a foil stop, time and proper time use the crossing fraction of the pre-step flight, consistent with the saved
thin-plane stop position. No within-foil slowing time is integrated; surviving crossings, IPA stops, boundary/decay
ordering, grazing paths and the low-energy model remain approximations requiring physics convergence checks.
A shared seed is not layout-invariant: chunking and compaction affect random draws.

## Bounded numerical convergence pilot

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.convergence \
  --g4 "$REF/g4ref.npz" --out "$DATA/convergence_UNIQUE_TAG" --n 2000 --seeds 11 22
```

This runs four arms sequentially for each saved seed: nominal (5 mm, 3000 steps, 4 loss slices), half-step
(2.5 mm, 6000 steps: equal maximum path), doubled cap (5 mm, 6000 steps), and 8 slices. `--geom` selects an overlay;
`--arms` can restrict the list. Put `nominal` first for paired arm differences. `run --max-steps` also exposes the
numerical cap directly; it is not a BO design variable.

`--start ROW --n COUNT` selects a contiguous whole-event block. Requests outside the table or splitting an event
(including noncontiguous repeated IDs) fail before creating output. The selected input is saved as `input_block.npz`;
the manifest records its hash, row range, event count and endpoint IDs. `nominal` must be the first arm. Disjoint
blocks in the same table are not automatically independent upstream source samples, and equal event IDs across
different G4 resampling jobs do not establish matched primaries.

The output directory must be new. Each arm retains an NPZ; `summary.json` counts actual saved muons, fates and foil
profiles and reports event-cluster standard errors for paired stop-fraction differences and flight-time shifts
conditional on stopping in both arms. `manifest.json` hashes the input, current beam/core physics code, field files
and geometry include closure. The fixed input prefix is reused; different seeds are response replicas, not new
primaries. Same seeds across layouts are not invariant common-random-number streams. No acceptance threshold or
independent-reference uncertainty is implied. Avoid interpreting conditional timing shifts as G4 stop-time closure.

`--budget-seconds` (default 1200) prevents starting further arms after the budget, but does not interrupt a running
arm; an incomplete pilot lists unstarted arms. On a GPVM use one CPU affinity and the one-job-per-host rule. All
passes include their compilation overhead and `repeat=0`: the legacy `muons_per_s_warm` metadata label from `run`
is **not** a warm-throughput measurement here. Never extrapolate larger-job runtime from this pilot.

## Finer slicing on a fixed population

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.slicestudy \
  --baseline "$DATA/convergence_UNIQUE_TAG" --out "$DATA/slices_NEW_TAG" --seeds 11 22
```

Requires a whole-event pilot with `input_block.npz`; pass the same `--geom` for moved targets. Reuses the saved
4/8-slice arms and runs only 16/32 slices. Before any run, verifies the selected-input hash, beam/core physics
source hashes, geometry include hashes, field hashes, and saved-arm parameters/lengths against the pilot manifest
and summary. This is not a full binary/environment identity check; reused output hashes are recorded at extension
time, not retroactive proof of their original write. The new output directory must not exist.

For each seed, all timing differences use the same intersection of tracks stopping in all four arms. Subsets keep
those stopping in the same foil across all arms separate from those changing foil in any arm. Saved row indices
allow exact replay. Reports include event-cluster errors for adjacent-slice and 32-minus-4 flight-time, foil-index
and z differences. These outcome-conditioned populations are **diagnostics**, not production response/correction
models; the common intersection changes if more arms are added. Seed replicas are correlated through their shared
inputs. No G4 time closure, acceptance threshold or default-slice change follows from the report.

## Frozen-entry mean-loss and layer-history diagnostic

```bash
PYTHONPATH=core:beam JAX_PLATFORMS=cpu python3 -m beamfastsim.layerstudy \
  --baseline "$DATA/convergence_UNIQUE_TAG" --out "$DATA/layers_NEW_TAG" --seed 11
PYTHONPATH=core:beam python3 -m beamfastsim.layerreport \
  --study "$DATA/layers_NEW_TAG" --slices "$DATA/slices_SAVED_TAG" --out "$DATA/layer_report_NEW_TAG"
```

Pass the same `--geom` to `layerstudy` for a moved target. On the GPVM, `pyenv ana` alone does not supply JAX;
use the existing project JAX environment (the path is documented in `art/run_fast_cpu.sh`), its `bin/python3`,
`JAX_PLATFORMS=cpu` and one CPU affinity. These are diagnostics, not BO parameters or a production-slice change.

`layerstudy` checks the saved whole-event pilot's source/input/geometry/field hashes, then observes its seed's
nominal arm with temporary, process-local ordered JAX callbacks. This deliberate replay is for histories absent
from the original output, not a replacement baseline. All six saved arrays (`x,p,t,fate,foil,nfoil`) must match
exactly before writing histories. The module restores the production functions and JIT cache after observation;
run it as a standalone process, not concurrently inside a shared interpreter. Callback timing is not throughput.

`layer_history.npz` records original row, chunk/step/lane, PRNG key, material/foil, path, entry direction and
momentum, exit momentum, and each sampled slice's entry momentum and requested loss. Rows remain aligned through
compaction/padding. First encounters with foil zero are frozen for deterministic 4/8/16/32/64/128-slice mean-loss
calculations. The numerical reference integrates the **same** stopping law from `T_STOP=0.1 MeV`, checks grid
refinement and reports critical entry energies at three path lengths. It does not integrate or certify the
physical range below the threshold. Out-of-domain critical-energy brackets fail explicitly.

`layerreport` reads these histories and the saved 4/8/16/32 arms without transport. It identifies coarse mean-loss
under-stoppers, records their sampled histories and saved outcomes, and tests threshold/grid sensitivity. All
entrance/outcome selections are diagnostics, not measured-variable response corrections or independent G4 entry
histories. The same-law range isolates integration error; it does not validate shell/high-order corrections,
muon-charge effects, straggling or thin-plane flight time. A finer fixed slice count is not automatically a
physically justified default.

## Release-produced loss tables and bounded integration

The existing Mu2eG4 `debug.storePhysicsTablesDir` hook can export the published release's tables without a build.
In a shell already set up with `muse setup SimJob MDC2025ac`, on an idle GPVM with one CPU affinity:

```bash
taskset -c 0 bash beam/art/export_physics_tables.sh "$REF" "$DATA/physics_NEW_TAG"
taskset -c 0 bash beam/art/export_physics_tables.sh "$REF" "$DATA/physics_NEW_TAG_b84" 84
taskset -c 0 bash beam/art/export_physics_tables.sh "$REF" "$DATA/physics_NEW_TAG_b168" 168
```

The reference directory must contain the saved expanded `job_config.txt` and `job.log`. The wrapper refuses an
existing output directory, the wrong published musing, or a nonempty saved `g4Macro`. It reads one original input
event, leaves its physics configuration intact except for enabling CSDA tables and the optional table binning,
and exports ASCII tables at endRun. A one-event run is necessary for art's managed Geant4 lifecycle; never inject
`/run/beamOn` into the macro. Do not run all three exports in parallel on one host. Outputs include the effective
FHiCL, macro, provenance hashes, log, small particle ntuple and all material/process tables. Check quota first.

In the project Python environment with `core:beam` on `PYTHONPATH`:

```bash
python3 -m beamfastsim.physics_tables --export "$DATA/physics_NEW_TAG_b168" \
  --compare-export "$DATA/physics_NEW_TAG_b84" --out "$DATA/physics_report_NEW_TAG"
python3 -m beamfastsim.rangestudy --layer-report "$DATA/layer_report_SAVED_TAG" \
  --table-report "$DATA/physics_report_NEW_TAG" --out "$DATA/range_NEW_TAG"
```

`physics_tables` reads the named `StoppingTarget_Al` material/cuts couple, not a hard-coded index. It checks the
ASCII schema/counts, density, couple/range-cut consistency and nonempty target vectors. Converted production-cut
energies in `cut.dat` are **keV**, unlike physics-vector energy in MeV. The reader is scoped to Geant4 11.3.2,
ShieldingM and Offline v10_47_00 and records source/table/configuration hashes; it does not certify historical
binary identity. Base-material-empty tables fail rather than being assigned a guessed substitute. The earlier
approximate G4 aluminium cut in the transport comment is not a substitute for the exported couple's actual cut.

`Ionisation`, aggregate restricted `DEDX`, unrestricted `DEDXnr`, `Range` and `CSDARange` remain distinct. The
between-node interpolant is a SciPy not-a-knot reconstruction, **not** a bitwise port of Geant4's spline routine.
Reports check dense-grid refinement and agreement between integrated reciprocal stopping power and native
`R(T)-R(0.1 MeV)`. Raw ranges include a below-grid tail; never compare them directly with a threshold-relative
fast range. Refinement reports include threshold-relative ranges and frozen-entry stop flags.

`rangestudy` selects negative-muon entries explicitly and reports excluded positive-muon row IDs. It reuses the
saved histories and compares the fast law, fast law at the exported electron cut, and G4 restricted/unrestricted
integrals. `adaptive_loss.adaptive_mean_loss` provides vectorized diagnostic Euler steps bounded by a requested
fraction of each lane's current kinetic energy. It reports actual steps, consumed path, fractional loss and
unresolved caps; it does not sample fluctuations or change transport defaults. The range study scans fractions
0.1 down to 0.001, checks energy/path errors as well as stop masks, and records G4 per-row outcome agreement.

Restricted continuous loss alone omits explicit delta rays and is **not** a full replacement for total loss.
Unrestricted mean loss is also not stochastic transport or projected range. These tests motivate a range-based
mean treatment with separately validated straggling/cut handling, not an automatic fixed-slice or fraction default.

## Frozen-entry range-mean fluctuation prototype (opt-in)

```bash
# Explicit PYTHONPATH is necessary: some site.env versions do not include beam/.
PYTHONPATH=core:beam JAX_PLATFORMS=cpu python3 -m beamfastsim.fluctstudy \
  --layer-report "$DATA/layer_report_SAVED_TAG" --table-report "$DATA/physics_report_SAVED_TAG" \
  --out "$DATA/fluct_NEW_TAG" --replicas 64 --seed 271828 --slices 4 16 32
PYTHONPATH=core:beam python3 -m beamfastsim.fluctreport \
  --study "$DATA/fluct_NEW_TAG" --out "$DATA/fluct_report_NEW_TAG"
```

Use the existing project JAX environment and `taskset -c 0` on an idle GPVM. No build, new G4 sample, production
transport replay or default change is involved. The input is the saved baseline-fast first-foil entrance selection,
with negative muons selected explicitly; it is not an independent G4 entrance sample. Each response replica reuses
these same rows. Positive muons remain excluded pending their own table/validation study.

### Mean and stopping contract

`rangefluct.sample_supplied_mean` traces the existing unjitted JAX Urban body with a temporary `_dedx` hook, then
restores the function. The supplied mean replaces only `S(T_in)*length`; physical length, entrance momentum,
material constants, cut and variance formula stay unchanged. The mean also controls the sampler's branch choice.
No core source or production API changes. **Standalone single-threaded interpreter only**: the trace hook is not
safe for concurrent sampler tracing. Original-mean equivalence, changed-mean Gaussian/Gamma/Urban moments, cap
bookkeeping, zero paths and deterministic range tests are covered by the scoped tests.

The range mode supplies `T_in - R_inverse(R(T_in)-h)` on each surviving slice. The tabulated range starts at
0.1 MeV, not zero. If `h > R(T_in)`, the diagnostic consumes all remaining energy **without fluctuations or hard
sampling**; exact threshold arrival is not a strict stop. This threshold-relative convention is inspired by G4's
range-exhaustion bypass, not an implementation of G4's physical range/lowest-energy cut. Already-subthreshold
entries and sampled subthreshold residuals are absorbed and recorded separately. Deterministic endpoints use the
inverse-range result directly to avoid a subtractive-rounding false stop. All range steps are corrected here;
G4 instead switches from local loss at a configured fractional-loss limit.

The local-mean arm is a new conditional response diagnostic with immediate threshold absorption and dense
interpolation of the original power, **not an exact stochastic replay of the production layer**. Comparing it to
range mode tests the mean/endpoint treatment with the law held fixed. The same-law arm cannot establish independent
low-energy physics, and a G4-tabulated mean does not turn the fast fluctuation kernel into G4 transport.

### Continuous/discrete delta-electron contract

| Arms | Continuous mean and sampler cut | Hard loss |
|---|---|---|
| `fast_local_1MeV`, `fast_range_1MeV` | Existing fast law, 1 MeV cut | None; any above-cut loss would be omitted |
| `fast_range_unrestricted`, `g4_range_unrestricted` | Fast unrestricted or G4 `DEDXnr`, cut effectively Tmax | None; inclusive diagnostic, not cut-matched G4 |
| `fast_range_cut_only`, `g4_range_cut_only` | Fast restricted or G4 aggregate `DEDX`, exported electron cut | **Intentionally omitted control**, not total energy loss |
| `fast_range_cut_split`, `g4_range_cut_split` | Same restricted range mean/cut | Separate analytic spin-1/2 Poisson delta spectrum above the cut |

In a split arm each slice freezes the hard rate/spectrum at its entrance, samples continuous loss, then subtracts
summed delta energy from the remaining muon energy with an explicit cap. Requested hard counts/energy, applied
energy, cap excess and threshold residual are separate saved arrays. Counts include requested transfers discarded
by a cap; they are not a transported-electron count. Electron kinetic energy is a muon energy debit, **not local
energy deposition**: no electron direction, transport, escape/deposition, bremsstrahlung or secondary response is
claimed. No unrestricted mean is combined with added hard deltas (which would double count).

The hard model is the core analytic spectrum, not release-specific G4 cross-section/scheduling parity. The report
compares its first moment with `DEDXnr-DEDX`; a small first-moment difference does not bound its distributional or
step-ordering error. Range-forced slices skip hard sampling entirely. Refinement, electron energy caps and the
threshold approximation must be examined before using this scheme in transport.

### Output and uncertainty

A new output directory is required. Inputs/core sources are hash-checked against the saved exact replay; all
consumed physics tables must be manifest-covered. Each arm saves a `(replicas, entrants)` NPZ with exit energy,
stop/forced-stop masks and an energy ledger. The manifest records source/input/output hashes and runtime versions;
source/input identities are rechecked at completion. `--budget-seconds` prevents new-arm launches, does not kill a
running arm, and incomplete studies exit nonzero. Seeds and replica layout must match for paired contrasts.

`summary.json` separates **conditional response MC SE** (replica totals) from the event-cluster sampling diagnostic
on per-entry averaged stop probabilities. Neither is a G4 reference/model uncertainty. Zero replica variation is
not zero physics uncertainty. `fluctreport` verifies every saved arm, energy ledger and stop count, then reports
stop-atom-inclusive CDFs, exit/loss quantiles, fixed entrance-energy cohorts (<3 MeV and >=3 MeV), within-entry
variance and paired CDF/energy differences. Survivor quantiles are outcome-conditioned and not substitutes for
unconditional comparisons. Neither flat counts nor matching exit means certify straggling distributions.

Before a transport A/B: validate hard/continuous distribution widths and range-endpoint behaviour, include mu+
separately, and declare stop-rate, first-foil, exit-energy and timing tolerances. Reuse transport baselines and keep
numerical same-law and physics G4-law candidates separate. This prototype is not a default-slice recommendation.

## Release-matched single-foil closure pilot

`foilclosure`, `foilcampaign`, `foilreport`, `foilcompare`, `foilendpoint` and `foilsaved` are opt-in diagnostics.
They use the published `ConstructEnvBox` and `Mu2eG4StudyReadBack` from SimJob MDC2025ac / Offline v10_47_00,
not a locally rebuilt module. Study geometry has no field and constructs NIST materials only; the foil is `G4_Al`
in a small vacuum world, not a renamed Mu2e material. No art output, build or production change is required.

In an idle one-CPU GPVM session with the published release set up, use an existing Python environment with
uproot/SciPy/JAX and explicit `PYTHONPATH=core:beam`. In the examples `$PY` is that environment's `bin/python3`;
all `$DATA` paths must be inside `/exp/mu2e/data/users/ram2aq/beam_fastsim`.

```bash
# Setup once in the session, never in a build/setup script.
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
muse setup SimJob MDC2025ac
export PYTHONPATH="$PWD/core:$PWD/beam" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# A new ten-event material/readout smoke with exported tables for both charges.
taskset -c 0 "$PY" -m beamfastsim.foilclosure --out "$DATA/foil_smoke_NEW" \
  --energy 5 --thickness .2 --events 10 --export-tables
taskset -c 0 "$PY" -m beamfastsim.foilreport --run "$DATA/foil_smoke_NEW" \
  --table-reference "$DATA/physics_report_SAVED" --out "$DATA/foil_smoke_report_NEW"
# Six cases, full/scattering-off/finer-loss-step arms, serially; fresh outputs only.
taskset -c 0 "$PY" -m beamfastsim.foilcampaign --out "$DATA/foil_matrix_NEW" --events 5000 --budget-seconds 900
taskset -c 0 "$PY" -m beamfastsim.foilcompare --campaign "$DATA/foil_matrix_NEW" \
  --table-export "$DATA/foil_smoke_NEW" --table-reference "$DATA/physics_report_SAVED" \
  --out "$DATA/foil_compare_NEW" --samples 50000 --slices 4 16 32
```

The material gate compares the exported named couples, density, cuts and five loss/range tables over 0.1–40 MeV.
Slab artifacts are checked against the export receipt; consumed negative-muon reference tables must be covered
by the saved reference manifest. Positive-muon comparisons without historical manifest coverage are explicitly
unqualified diagnostics. Matching stopping-power tables alone does not prove identical fluctuation parameters.

### G4 model isolation and scoring

- `full` leaves ShieldingM physics intact. `loss` disables **`msc` and `CoulombScat` for mu- only**; other processes,
  secondary production/transport, and discrete-ionisation recoil remain active. It is not strictly straight-line
  transport. `loss_fine` additionally sets `/process/eLoss/StepFunctionMuHad 0.05 0.001 mm`.
- The process commands require Idle state. The diagnostic pre-init macro calls `/run/initialize` after user actions
  are registered, applies the changes and returns to art; it never calls `/run/beamOn`. The subsequent normal
  Initialize call does not reconstruct already initialized geometry/physics. Log echo and absence of primary
  scattering step codes are checked. This is a study-specific hook, not a production configuration pattern.
- The release's visualization macro hook is compiled out and does not execute these commands. `muMsc` is also
  not the installed process name in this physics list. Both failures can leave art exit 0; illegal/aborted macro
  messages fail the diagnostic. Never accept exit 0 alone or reuse a rejected smoke directory.
- `readback/tpart` and `readback/ntstepper` use their **highest ROOT cycles**. One primary of the requested charge
  with creation code 56 and every requested event ID must be persisted. The first outside pre-step state **after
  foil entry**, matched by event/track ID, gives exit energy; the final SimParticle state in vacuum does not.
- `isstop` only indicates a known end volume in this analyzer. A physical foil stop instead requires zero endpoint
  momentum, in-foil endpoint and a defined decay/capture stopping code. Undefined endpoint sentinels are unresolved,
  not stops. Subthreshold exits remain separate from physical stops. Comparator threshold alignment collapses
  exit energies below 0.1 MeV to zero explicitly, not retroactively to a physical G4 stop.
- Delta electrons are direct children of the primary with creation code 31 (`muIoni`), born in the foil before
  first exit. Their initial kinetic energy is a primary-energy debit, never local deposition. Direct brems gamma
  energy is counted separately. Remaining primary loss is labelled **nonhard muon loss**, not deposited energy.
  Pre-step chord sums are geometric diagnostics, not Geant4 true path length. The readback lacks true step length
  and deposited energy; a future deposition/timing closure may need a richer dumper and a user build. Saved
  `layer_time_ns` for physical stoppers includes the at-rest decay/capture wait, not just slowing time; no timing
  closure is assigned from this quantity.

### Refinement, endpoint sensitivity and final saved-data report

```bash
# No rerun of the original G4 arms: add dense-table and finer-step edge diagnostics.
taskset -c 0 "$PY" -m beamfastsim.foilcampaign --out "$DATA/foil_refine_NEW" --events 5000 \
  --cases nom_edge moved_edge oblique_edge --refine --bins 168 --budget-seconds 900
# Separate endpoint sensitivity; no change to the supplied-mean prototype or production transport.
taskset -c 0 "$PY" -m beamfastsim.foilendpoint --comparison "$DATA/foil_compare_NEW" \
  --table-export "$DATA/foil_smoke_NEW" --out "$DATA/foil_endpoint_NEW" --samples 50000
# Final comparisons reuse every saved fast/G4 array, including highest-cycle G4 refinements.
taskset -c 0 "$PY" -m beamfastsim.foilsaved --comparison "$DATA/foil_compare_NEW" \
  --refinement "$DATA/foil_refine_NEW" --endpoint "$DATA/foil_endpoint_NEW" \
  --table-export "$DATA/foil_smoke_NEW" --table-reference "$DATA/physics_report_SAVED" \
  --out "$DATA/foil_final_NEW"
```

The refinement adds step settings `(0.05, 0.001 mm)`, `(0.01, 0.0001 mm)` and `(0.005, 0.00005 mm)` at 168 bins/decade.
Compare the first to the saved 7-bin fine-step arm for table-resolution sensitivity, then the latter two for step
sensitivity. Fresh seeds are independent response samples, not matched stochastic trajectories. The raw-range
endpoint arm uses native Range/CSDARange, including its subthreshold offset, but bridges its inverse below 0.1 MeV
linearly and still absorbs sampled subthreshold residuals. This is an **unvalidated endpoint sensitivity**, not a
physical-zero-energy prescription or a replacement baseline.

Provisional pilot bands are 0.02 absolute stop/CDF, max(1% mean loss, 2 keV) for mean energy, and 5% for variance.
They are **not agreed chain acceptance**. Stop intervals are conservative 99% exact-binomial difference intervals,
including zero/all-stop boundaries. Moments use pointwise 3-SE approximations; CDFs use conservative 99% two-sample
DKW brackets. Verdicts are PASS/FAIL/**INDETERMINATE**; absence of a resolved difference is not equivalence. With
5,000 G4 and 50,000 fast samples the DKW radius is 0.0322, so a 0.02 CDF PASS is impossible at this pilot precision.
Reports are not multiplicity-adjusted. `full-minus-loss` isolates the configured scattering change;
`full-minus-loss_fine` also changes stepping and must not be labelled a pure scattering effect.

`foilsaved` audits original ROOT/census identity and energy ledgers, uses the corrected intervals, checks all
consumed slab tables, records current dependencies and hashes its final summary. Earlier candidate reports remain
historical; source and summary hashes are not retroactive binary certification. Do not promote a fast mean law,
endpoint or slice count until the physical-stop/threshold distinction, G4 discretisation and shape precision are
resolved, and validate mu+ separately before a mixed-charge full-chain A/B.

Validation status is diagnostic only; neither table agreement nor aggregate stop counts establish physics closure.

## Precision single-foil cells, mu+ tables and transport arms (opt-in)

**Tables.** `beamfastsim/data/g4_target_al_mu.npz` (StoppingTarget_Al of the full geometry, the transport default
of the `g4` law) and `g4_al_mu.npz` (G4_Al study slab) hold the 168-bin/decade CSDA range, unrestricted dE/dx and
msc transport mean free path of both charges (`python3 -m beamfastsim.g4tables EXPORT/tables --material M --out F`).
mu- tables are identical. mu+ are not: below 0.2 MeV `G4BraggModel` takes PSTAR data by material name, which matches
`G4_Al` but not the `StoppingTarget_Al` alias, and the Bethe dE/dx above is joined to it with a `0.2 MeV / T` term
(slab/target -2.1 % at 0.2 MeV, -0.44 % at 1 MeV). Single-foil closure therefore uses `table="slab"`; the transport
uses `table="target"`. The `tables` grid task can re-export full-geometry tables from a one-event gun job;
its output archive is not evidence of success without a checked completion marker and table contents.

**Layer physics presets** (`transport.params`, CLI `--physics/--mean/--law/--msc-path`): `legacy` remains the default
(local mean, fast Bethe law). `g4` is an **unvalidated diagnostic**: range-integrated slice means plus a deterministic
MSC effective-path approximation. It is not Geant4 transport and is not eligible for production promotion from
aggregate stop counts. `msc=False` disables both the angular kick and effective-path correction. Requires the
separate core `eloss_jax.sample_loss(..., mean=)` change; select that checkout explicitly on `PYTHONPATH`.
The campaign's companion checkout is `../beam-core` relative to this worktree; the umbrella's empty `core/`
submodule directory and an older main core do not supply that API.

**Grid** (`art/submit_beam_closure.sh STAGE TASKS [--test] [--submit]`, worker `art/grid_beam_closure.sh`, code
tarball staged on pnfs, `BEAMFS_CORE` = the core checkout, `MEMORY` overrides 4000MB). Task lines:

```text
foilscan SCAN PDG [EVENTS [SEEDOFF]] MODES...   G4 slab runs (foilscan.SCANS: flat scans t1056/t2/t4 or cells)
tables LABEL GUNFILE BINS GEOM                   one-event full-geometry physics-table export
arms LABEL G4REF START N GEOM ARMS...            beamfastsim.convergence transport arms (JAX from LCG_108)
chain LABEL GUNFILE SKIP N SEED GEOM             G4 DS-stage reference on a gun list (beamfastsim gunfile)
```

**Comparisons** (laptop or GPVM, JAX): `python3 -m beamfastsim.foilcells --campaign DIR... --slab-tables
SLAB_EXPORT/tables --out NEW` scores the monoenergetic cells (threshold-relative `rangefluct` arms and the production
layer arms of `foilfast`) against G4 `loss_finest`, `full_finest` and stock `full`, and applies the endpoint
contract: physical stop (range-out in the foil, G4 zero momentum in the foil) and subthreshold exit (0 < T < 0.1
MeV) are scored separately against a 0.02 band with exact-binomial 99 % intervals. `foilscan compare` gives stop-edge
maps (T50 per cos bin) for the flat scans. `convergence --arms nominal slices32 range_fast range_fast_s32 range_g4 g4`
gives paired whole-chain stop and first-foil differences per charge on a whole-event block.

### Audit and recovery safeguards

`foilreport.audit` reads projected branches in 16 MB chunks with the ROOT array cache disabled. It retains
per-primary reductions rather than the entire step tree, including across chunk boundaries and secondary-only
chunks. Both trees use their highest cycles. Exact hard-electron energy quantiles use temporary disk storage
(8 bytes/electron, under `TMPDIR`, automatically removed). The eager census remains a test oracle; chunk-size,
identity and floating-point equivalence tests cover the streaming path.

Table packaging requires **explicit output and material**, refuses overwrites, hashes both charges' consumed
loss/range tables, metadata and MSC input, and validates arrays/material on load. Existing packages without hashes
are labelled unqualified. A fresh export can be compared without replacing the package:

```bash
python3 -m beamfastsim.g4tables compare "$FRESH_EXPORT" beam/beamfastsim/data/g4_target_al_mu.npz \
  --material StoppingTarget_Al --require-audited-export > "$DATA/table_comparison.json"
python3 -m beamfastsim.pilotaudit --pilot "$PILOT" --g4 "$ORIGINAL_G4" --out "$DATA/pilot_audit_NEW.json"
```

`pilotaudit` verifies original input and selected-block hashes, every original column at the recorded row offset,
whole-event selection, saved counts/metadata/summaries and event-cluster paired statistics. It separately reports
zero-momentum and positive subthreshold `fate=1` endpoints. The latter are **not physical-rest stops**, even though
the legacy chain's fate convention counts them. Raw zero momentum also does not prove correct range-out physics.
No historical binary certification or physics acceptance is inferred from artifact hashes.

`foilcells --charges -13 --cells moved_edge` supports an explicitly scoped positive-muon cell. Every requested
cell/charge requires loss-only, refined full and stock full references; duplicate sample/seed identities and
missing references fail. Consumed slab tables are checked against an audited export. All fast response arrays,
source/core/table identities and comparisons are saved. A `complete` report means the requested diagnostic ran,
not that acceptance passed. A 5,000-G4/50,000-fast pilot cannot establish the 0.02 CDF band (99% DKW radius 0.0322).

The grid submitter's dry-run does not stage, delete or contact the grid. Real submissions require a fresh staging
directory and explicit user approval. Workers propagate expansion/application/table-export failures and publish
no success archive on failure. A completed archive receives an adjacent `.ok` hash marker only after copy-out;
an existing archive without a marker is refused, not silently skipped or overwritten. Old failure archives must
be audited separately. These guards do not replace highest-cycle event census or physics validation.

### Stochastic slab-path diagnostic

`slabdiffusion` is an opt-in infinite-plane first-passage model, not a change to beam transport. It advances a
polygonal path, samples tangent angular increments with per-axis variance `ds/lambda1(T)`, and intersects both
faces. It neither inverts an ensemble mean projected path across a foil nor adds a final Highland kick.
It matches the angular-diffusion first moment, not the G4 single-scattering tails or boundary algorithms.
The release source anchors are Geant4 v11.3.2 `G4WentzelVIModel::{ComputeTruePathLengthLimit,
ComputeGeomPathLength,ComputeTrueStepLength}` and `G4VMscModel::GetTransportMeanFreePath`. G4's path conversion
belongs to step-limited segments; using it as a whole-foil traversal condition is not equivalent.

```bash
PYTHONPATH=../beam-core:beam python3 -m beamfastsim.slabstudy \
  --cell moved_edge --pdg -13 --samples 2000 --seed 71 \
  --arms no_scatter diffuse half_step half_angle half_energy refined \
  --slab-tables "$SLAB_EXPORT" --campaign "$SAVED_G4_CAMPAIGN" --out "$DATA/slab_NEW"
```

The host API accepts explicit tables, material, row/replica IDs, step controls and energy floor. Response arrays
include consumed path, exit direction/position, residual energy, endpoint status and an energy ledger. Keys depend
on row, replica, step and channel, not host batch boundaries. Numerical refinement is identity-paired but not
Brownian-bridge coupled. Floor/cap/numerical endpoints stay unresolved; sampled-loss clipping is separately labelled
raw zero, not certified physical rest. The floor has a 16-ulp-relative allowance to avoid a zero-range-budget loop;
residual energy is retained, never snapped or debited. Lowering the floor only within the supplied table domain is
allowed. Threshold-identifiable counts may be reported, but unresolved populations cannot establish physical-stop
or ordinary exit-energy-distribution closure. No production promotion follows from a stable aggregate pilot.

`slabendpoints` retains all entries and bounds a floor/cap continuation by `[0, residual energy]`, conditional on
monotone primary-energy debit; numerical failures use `[0, entry energy]`. RAWZERO is singleton zero only under the
algorithmic diagnostic convention, never a certified physical stop. The explicit `E < 0.1 MeV -> 0` transform can
be fully identified even when physical rest versus subthreshold escape is not. `slabstudy` compares this transformed
distribution only when all transformed intervals collapse; an ordinary endpoint distribution remains unqualified.
Raw-energy CDF envelopes separate identification ambiguity from sampling uncertainty. Face distances are normal
geometric distances, not remaining stochastic ranges; CSDA range is not used to claim escape is impossible.

The runner accepts one or more saved `foilscan` or legacy `foilcampaign` directories. It checks selected geometry,
charge, seeds, receipt/artifact/array identities and event census, re-auditing legacy ROOT trees with the streaming
reader. Original mode labels are retained: stock `full` is not renamed converged. Axial-face validation rejects
reference exits incompatible with the infinite-plane model. Mu+ slab validation does not certify target-material
mu+ response, and the shared charge assumption for the MSC table remains unvalidated.

```bash
PYTHONPATH=../beam-core:beam python3 -m beamfastsim.slabstudy \
  --cell moved_edge --pdg 13 --samples 2000 --seed 72 --arms no_scatter diffuse refined \
  --max-steps 20000 --slab-tables "$SLAB_EXPORT" \
  --campaign "$SAVED_MATRIX" "$SAVED_REFINEMENT" --out "$DATA/slab_NEW"
```

On a GPVM, restrict the process to one CPU (for example `taskset -c 0` on a host whose allowed CPU list includes 0)
as well as setting BLAS/OpenMP thread counts. Set `JAX_PLATFORMS=cpu` for a CPU diagnostic. A refined low-energy arm
can exhaust the default 10000-step cap; inspect statuses and run an explicit cap sensitivity rather than relabelling
those histories. Stable threshold counts alone do not establish convergence of physical endpoints.

To extend reporting without regenerating responses, add `--responses "$SAVED_SLAB_STUDY"` with the same cell,
charge, sample count, seed, arms and numerical settings, and a new output directory. Historical source provenance
stays in the old study, separate from the new reporting sources. All saved input/output identities are rechecked.
This is reporting, not a new timing measurement of the kernel.

Neither the whole-foil effective-path preset nor this diffusion/CSDA-mean finite-cut fluctuation hybrid is a
validated replacement. A passing mean/threshold check can coexist with a failing width or tail distribution.
A consistent restricted-soft/explicit-hard loss treatment, independently validated low-energy endpoints, angular
and boundary validation, and real production boundary/displacement integration are separate requirements.
`chaincompare` retains exploratory first-prefix/row-level statistics; use `pilotaudit` for the saved whole-event
pilot instead. No acceptance claim should be based on its exploratory metrics.

### Restricted-soft / explicit-hard slab diagnostic (opt-in)

`slabstudy --loss-model split` selects the audited charge-specific **restricted
Range**, at the export's electron production cut, instead of inclusive CSDARange.
It passes each range-integrated soft mean to the Urban sampler and independently
samples analytic spin-1/2 compound-Poisson transfers above the cut. The companion
core module `mu2efastsim.eloss_split_jax` is required. Existing slab `hybrid` and
all production transport defaults are unchanged.

```bash
PYTHONPATH=/path/to/core:beam JAX_PLATFORMS=cpu python3 -m beamfastsim.slabstudy \
  --cell moved_5 --pdg 13 --samples 5000 --seed 91 \
  --arms no_scatter diffuse refined --loss-model split --max-steps 20000 \
  --slab-tables "$SLAB_EXPORT" --campaign "$SAVED_MATRIX" --out "$DATA/split_NEW"
python3 -m beamfastsim.slabgate --study "$DATA/split_NEW" --out "$DATA/gates_NEW.json"
```

The hard spectrum is frozen at leg-entry momentum. `energy_fraction` bounds the
**soft mean**, not the full stochastic debit; step refinement remains necessary.
There is no discrete recoil, electron transport, deposition, or new angular model.
Soft and hard requested/applied energies and requested hard counts are saved
separately; soft is debited first. Hard counts include transfers discarded by
kinetic clipping. Floor/cap histories retain positive residuals and stay unresolved;
RAWZERO is an algorithmic clamp, never physical stopping.

The inherited Bohr fluctuation branch has not been validated as a restricted
variance when `cut < Tmax`. Split sampling **fails closed with a nonfinite soft
request** in that overlapping-hard branch (mean >= 10*cut and Tmax <= 2*cut), and
the slab records an unresolved numerical endpoint without applying either debit.
No-hard branches and legacy sampling are unchanged. Urban plus the hard spectrum
remains a model requiring independent distribution gates, not exact G4 parity.

The runner records restricted-range hashes, material, charge, cut and actual table
arrays. Split response reuse rejects absent or different transport bindings, and
historical response-generation provenance stays separate from current reporting.
A mean audit checks restricted power plus analytic hard power against inclusive
power over 0.1–40 MeV. A second audit compares the derivative of the actual log-log
range interpolant with reconstructed restricted power down to the table floor.
These are diagnostics, not calibrated tolerances. The bare `respond` API's
`range_kind='restricted'` is a caller assertion; an unannotated table alone cannot
verify its material/cut contract.

`slabgate` verifies saved input/output hashes without requiring historical source
files to remain unchanged. It selects the finest *available labelled* loss
reference for `no_scatter`, and full-scattering reference for other arms; it never
silently calls stock full converged or uses full-minus-loss as closure. It retains
all-entry threshold-transform intervals, unresolved counts and actual reference
labels. There is no global-pass or physical-stop flag. Reference byte/cell/census
checks do not yet compare material/cut/process signatures semantically; reports
mark that qualification **UNVERIFIED**. Slab mu+ does not certify target-material
mu+, and charge equivalence of the shared mu+-derived scattering table is still
unverified. A small CDF point estimate is not equivalence: the independent
reference size must permit its uncertainty band to fit inside the tolerance.

### Reduced-reference recovery and precision planning

`lossdiagnostic` is a read-only, opt-in saved-data reporter, not transport or a
reference generator. Use it when only the historical `foil_threadG_compare_*`,
`foil_threadG_final_report_*` and slab ASCII tables remain:

```bash
PYTHONPATH=/path/to/core:beam python3 -m beamfastsim.lossdiagnostic \
  --recovered-root /path/to/recovered/beam_fastsim --out /path/to/NEW.json
```

The reporter checks the canonical summary hash, refined-array hashes when
covered, persisted event/cell/outcome census, moments and consumed table hashes.
Original reduced arrays lacking their generation manifest are explicitly only
current-byte and historical-reduction checked. Missing effective configs, logs,
macros, ROOT and receipts keep reference configuration **UNVERIFIED**; a case
label or a packaged table provenance record cannot replace them. Old reduced
arrays lack entry cosine and exit direction: recorded incidence is not a measured
angular distribution. Chord sums are not true G4 path lengths. Source/table
identity is not historical binary or physics certification.

The all-entry `T < 0.1 MeV -> 0` variance decomposes into positive-component within
variance plus between-zero/positive variance; the positive component is
outcome-selected. Population components sum to population variance; the report
also gives unbiased sample variance and pointwise influence-function SEs. This
decomposition does not attribute a fast/reference difference to scattering.
Unknown one-sided endpoint residuals in `[0,floor]` change population variance by
at most `floor*sqrt(variance) + floor**2/4`; that diagnostic bound neither
identifies physical stopping nor verifies unavailable response arrays.

Precision planning retains both independent 99% DKW radii. At a 0.02 CDF band,
5,000 reference entries alone already give radius 0.02448. Zero measured distance
requires at least 29,958 entries **per arm** for equal samples; reserving 0.01 for
measured distance requires 119,830 per arm. With one million fast entries, the
report computes the independent reference size with the same headroom. These
are pointwise feasibility calculations, not a predicted PASS, simultaneous
certificate, runtime estimate or permission to generate G4.

The report also inventories charge-specific material tables and the unsupported
overlapping-hard Bohr region. A frozen-power leg-length screen is not a count of
actual transport histories. Slab mu+ cannot certify target-material mu+; sharing
a mu+-derived scattering table across charges remains unverified. No production
setting or acceptance gate changes.

The variance-size proxy scales the observed pilot fourth-moment SE to a specified
3-SE relative budget (default 2.5% of reference variance, reserving part of the
5% diagnostic band). This asymptotic planning proxy is not a tail guarantee or a
power calculation. The unsupported-Bohr screen also evaluates the documented
0.01-mm / 5%-energy leg ceiling on the actual log-log restricted Range grid;
no-scattering and omitted angular/face limits make this a conservative leg-size
screen, not an execution trace. Refinement reduces that ceiling. Neither check
admits a new physics regime or changes the split sampler's fail-closed guard.

### Follow-up on recovered artifacts

`lossfollowup` runs **new** slab response samples against lower-tier recovered
reductions. It is separate from the audited `slabstudy` path, whose checks are
unchanged. It requires the recovered table bytes and exact numerical equality
with the supplied slab package, and records that the original reference
configuration remains unverified. It saves all response arrays, keyed identities,
ledgers, unresolved endpoints and threshold-mixture comparisons. Never describe
its new outputs as recovered historical arrays or qualified G4 closure.

```bash
PYTHONPATH=/path/to/core:beam python3 -m beamfastsim.lossfollowup \
  --recovered-root /path/to/recovered/beam_fastsim --cell moved_edge \
  --samples 2000 --seed 92 --arms no_scatter diffuse refined --out /path/to/NEW
python3 -m beamfastsim.lossexplain --study /path/to/NEW --out /path/to/NEW_explanation.json
```

`lossexplain` decomposes the all-entry threshold variance difference exactly into
symmetric occupancy, conditional-mean-squared and conditional-width terms with
correlated delta-method SEs. Those are algebraic contributions, not mechanisms.
The loss-channel covariance budget uses applied energies after path/stop
selection; it is not a primitive microscopic variance estimate. Keyed numerical
refinements retain their matched sample identity, not Brownian-bridge coupling.
Independent seeds must not be paired as refinements.

`lossarchive` audits a recovered original grid scan bundle and the saved
whole-event pilot, preserving missing ROOT/energy-cut-export limits. Its
`--full-fast/--full-metrics` option checks an older full baseline's counts and
profiles, but explicitly does not infer historical row alignment absent a saved
input hash. `lossscan` selects a uniform outcome-independent subset of the archived
scan, uses actual entry energy and incidence, and retains every selected event
in fixed energy/angle bins. It compares paired threshold outcomes, not physical
stops or exit angles. No command generates G4 or changes production defaults.

### Bounded refined-full / lateral-displacement edge controls

`foilclosure --lateral-displacement on|off` is explicit and opt-in; omission
preserves the inherited setting. `edgeg4 prepare` writes a new, bounded mu−
reference directory (at most 5k events) without running G4. Both named modes
use 168 bins/decade and StepFunctionMuHad 0.005 / 0.00005 mm; only the lateral
boolean differs. Use independent seeds. A single refined level is not a
convergence proof, and old stock references retain their original qualification.

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.edgeg4 prepare \
  --out "$DATA/NEW" --cell moved_edge --mode full_refined --events 5000 --seed 12345
# In a Slurm CPU allocation, with the saved EL8 client and pinned stock EL9 worker image:
export CVMFSEXEC_DIST=/path/to/saved/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:aa/REPLACE_WITH_PINNED_IMAGE
export EDGE_MOUNT_RUNNER=/path/to/container/offline/offline_uva.sh
bash beam/art/edge_offline_uva.sh "$DATA/NEW"
# Back in the analysis Python environment:
PYTHONPATH=core:beam python3 -m beamfastsim.edgeg4 finalize --out "$DATA/NEW"
```

The opt-in runtime adapter uses the unchanged Offline UVA wrapper's mounted-image
entrypoint, a unique node-local cache and the already-published MDC2025ac release;
no build, install or image production. Finalization checks prepared/source hashes,
effective configuration, echoed physics macro commands, completed event count,
run-specific table export and highest-cycle streaming ROOT census. ROOT is retained.
`edgeangles` adds first-exit directions/positions from the same existing primary
pre-step rows, identity-matched to that census. Stoppers keep NaN directions;
chords are not true G4 step lengths. `edgecompare` reuses finished fast arms and
reports independent comparisons and outcome-conditioned angular diagnostics;
it never reruns a sampler or silently certifies physical stopping.

### Saved edge-shape and local angular-increment diagnostics

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.edgeshape \
  --root "$EDGE_REFERENCES" --prior "$EDGE_COMPARISON" \
  --study "$NORMAL_COMPLETED_FAST" "$OBLIQUE_COMPLETED_FAST" --out "$DATA/NEW_SHAPES"
PYTHONPATH=core:beam python3 -m beamfastsim.edgemoments \
  --root "$EDGE_REFERENCES" --out "$DATA/NEW_LOCAL_MOMENTS"
```

These commands read completed, hash-bound references through the unchanged
strict loader and stream the original highest-cycle readback. They run no G4
or fast sampler. `edgeshape` retains all entries in fixed diagnostic energy
caps and exact variance partitions, adds exit-energy/angle standardization,
and records primary-identity-bound adjacent angular/chord metrics before the
first exit. `edgemoments` clusters local angular moments by primary against
exported lambda budgets, including a q<=0.001 screen; it never treats millions
of correlated steps as independent events. Neither command certifies physical
stops for positive-floor fast endpoints or invents missing true G4 step lengths.
Output-conditioned and capped statistics do not replace original gates. See
[edge diagnostics](docs/loss-edge-diagnostics.md) for numerical qualifications.

`python3 -m beamfastsim.edgeblocks --root "$EDGE_REFERENCES" --out "$DATA/NEW_BLOCKS"`
aggregates the same retained primary rows into fixed 0.001/0.005/0.01-mm
chord bins, retaining each block's actual endpoint deflection and summed
chord/lambda budget. This saved-trajectory diagnostic checks scale sensitivity
without generating a trajectory or claiming true-path/fast-leg equivalence.
The bounded per-file primary projection is retained in memory; use a compute
allocation, not a login node.

### Explicit step/bin edge factorial

`edgefactorial` adds three opt-in lateral-on modes without changing `edgeg4`'s
existing two modes: `stock_step_bins7`, `stock_step_bins168`, and
`refined_step_bins7`. Stock step means no step-function override in the released
runtime; it does not certify an unavailable historical configuration. Refined
step is 0.005 / 0.00005 mm. The export-bin setting is explicit, so a 7-bin arm
never silently inherits dense export. Existing defaults remain byte-identical.

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.edgefactorial prepare \
  --out "$DATA/NEW" --cell moved_edge --mode stock_step_bins7 --events 5000 --seed 12346
# Execute the prepared directory using the same private-cache adapter above.
PYTHONPATH=core:beam python3 -m beamfastsim.edgefactorial finalize --out "$DATA/NEW"
PYTHONPATH=core:beam python3 -m beamfastsim.edgefactorial collect --root "$FACTORIAL_REFERENCES"
PYTHONPATH=core:beam python3 -m beamfastsim.edgefactorialreport \
  --root "$FACTORIAL_REFERENCES" --reused-root "$EDGE_REFERENCES" \
  --study "$NORMAL_COMPLETED_FAST" "$OBLIQUE_COMPLETED_FAST" \
  --recovered-root "$RECOVERED_REFERENCES" --package "$SLAB_PACKAGE" --out "$DATA/NEW_FACTORIAL_REPORT"
```

Finalization additionally verifies the actual native table-node density and
run-specific material/cuts. Reporting reuses the completed refined/168 arm,
reaudits the original highest-cycle ROOT counts, checks identical runtime
binary/source hashes and keeps independent seeds distinct. Simple step/bin
contrasts, averaged main effects and interaction estimates retain shared-arm
covariance; related estimates are not independent or simultaneous acceptance
tests. Two numerical levels do not prove convergence. Coarse exported tables
are separately audited, not required to equal the dense fast-response package,
and diagnostic spline comparisons do not claim Geant4 interpolation parity.
