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
