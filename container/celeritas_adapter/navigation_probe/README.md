# h7a geometry-aware field navigation probe — corrected preparation

This isolated target uses the **existing user-built h5 Celeritas 0.6.3 installation** and its VecGeom/G4VG
libraries. It does not change h5/h6, the container, or production physics. The user-built h7a executable
passes the revised synthetic navigation suite across two retained runs. Real Mu2e/material/scoring and
performance qualification remain open (see follow-up below).

## Observable and interface contract

The standalone probe loads public GDML through Geant4 and G4VG, then uses the actual host/device
`VecgeomTrackView`, Cartesian field, Dormand–Prince integrator, `FieldSubstepper` and `FieldPropagator`.
Each independent trajectory has particle state for charge/momentum, but **no physics interactions, sensitive
hit filtering, energy loss, scattering or deposited-energy scoring**. Every propagation call and boundary
transition is recorded, including boundaries with zero deposition.

Records contain pre/post world positions, directions, public transport momentum, requested and reported
path, accumulated path/time, logical and placed-volume IDs before and after crossing, on-boundary/outside
states, substepper/field-evaluation counts, public looping flag and completion/status markers. Logical and
physical label tables accompany results. No caller-side push or implicit retry hides lack of progress.

**Version-specific limitations:** VecGeom `normal()` is unimplemented; fixture normals are independent
analytic checks, not claimed as navigator output. `failed()` is always false in this backend. A propagator's
internal momentum and inner iteration-exhaustion status are private. Public momentum magnitude is conserved
by construction; it is not an independent integration-drift measurement. The recorded path is the
propagator's reported curved distance, not a recomputation from endpoint chords. A public `looping` flag
means outer `max_substeps` exhaustion, not proof of a closed orbit. Terminal loop/budget failures remain
failures except in separately declared negative fixtures.

Every Runge–Kutta trial field query is guarded. An uncovered/nonfinite query aborts the **whole process**
(host exception or device trap before interpolation), rather than letting an invalid state reach geometry
or accepting a zero/clamped substitute. Deliberate guard fixtures run separately per backend and are
negative process tests, never accepted transport records or sanitizer-clean positive cases.

## Preparation and validation

`fixtures.py` generates public box/slab geometries and explicit mm/T/MeV-c requests with independent
straight-line/helix or DOP853 references. Checks distinguish numerical residuals from statistical uncertainty.
Thin layers, repeated crossings and near-tangent hit/miss paths are tested; exact tangency is not assigned an
arbitrary side convention. `prepare.py` writes a new source snapshot plus pinned h4 parser and source/API
provenance without configuring or compiling. The prior h6 source remains untouched.

Eight fixture cases contain 24 independent path requests: zero/constant/affine fields, repeated 0.1/0.01 mm
slab crossings, near-tangent hit/miss pairs, world exit, explicit outer-substep and record-budget exhaustion,
and separately executed map-guard aborts. A neutral control crosses the nonzero affine map. An oracle-only
regression requires the affine gradient's effect to exceed the acceptance envelope for replacing it with a
constant field. The runner hashes sanitizer logs; replay rechecks their summaries, not just a boolean.
No exact-tangent side convention, surface normal from VecGeom or physics accuracy claim is supplied.

Preparation validation at h7a handoff: **95 local Python tests pass, plus 30 parameterized subtests**, with the pinned
upstream preparation test and saved h6 replay enabled. These are source/contracts/oracles, not h7 transport.
The exact installed h5 API header hashes match the reviewed v0.6.3 source. Source-level reviews corrected a
namespace closure and restricted small-step stall detection to incomplete non-crossing requests; no compiler
or runtime success is claimed.

## Initial build failure and isolated correction

The user's October 27 h7 build failed in `probe.cc`: five `vector_json(..., millimeter)` calls passed a
`celeritas::Constant` to a `double` parameter without the required explicit conversion. The JSON initializer
errors were downstream diagnostics. All five now use `static_cast<double>(millimeter)`, as in the existing
h6 serializer. Floating-point arithmetic with the unit constant uses Celeritas's supported overloads and is
unchanged. A source regression check covers all five calls; it is not a substitute for compilation.

The saved build log and CUDA object show that `NavKernels.cu` compiled with `-rdc=true`; no executable or final
device-link result was produced. Preserve `celer_navigation_h7` and `celer_navigation_h7-build` as the original
snapshot and failed build. The corrected source below uses **new h7a source and build directories**. Fixtures,
budgets, navigation algorithms, CUDA source and dependencies are unchanged. No corrected C++/CUDA build has
been run by the agent; the user subsequently completed the build as recorded below.

## User-build and initial runtime follow-up

The user-built h7a binary has SHA256 `067a3c101c2de4ab392549c71fb08a021834c27615a0b599fca222a1daf5cf18`.
Source/parser/cache bindings, RDC, explicit CUDA device linking and fresh h5 library resolution were checked.
The first runtime's affine and constant cases pass independent host/device replay, with clean memcheck and
all expected crossing sequences; both deliberate guard processes fail with the required marker and no result.
The full run **fails**, however: outer_limit hits the map guard before returning its expected exhaustion
status. The remaining four cases were not reached. Original outputs and the failed full replay are retained.

Only the outer_limit fixture's max_step_mm is now 4 instead of 100. Length, max_substeps=1, geometry/map and
all acceptance budgets are unchanged. The initial Dormand–Prince stage-position envelope, bounded by the
absolute Butcher-row sums times the cap and unit-speed RHS, fits the initial 175 mm map margin; the cap's
arc sagitta still greatly exceeds delta_chord. This tests outer exhaustion without conflating it with the
separate map-domain negative. No binary/source snapshot modification or rebuild is required.

The checker supports `prepare --case NAME` and `check --case NAME`, repeatable. Selected sets/replays are
explicitly marked as not full-suite passes; result/log/sanitizer/fixture hashes and fixed oracles remain
checked. Runtime continuation runs only corrected outer_limit plus the unrun record_limit, tangent,
world_exit and zero cases, reusing saved affine/constant/guard evidence. The continuation completed on an
A100-80GB in 11 s, with clean memcheck and passing independent replay for all five cases.

`report.py ORIGINAL_RESULTS CONTINUATION_RESULTS` replays both selections, verifies identical executable,
source, runner and library identities, regenerates and byte-compares the revised fixtures, requires exactly
the documented cap change, and checks exhaustive/nonduplicated case coverage. It reports the corrected
suite pass separately from the original failed run; no records or manifests are merged or rewritten.

Combined result: **8 fixture cases; 42 positive host/device trajectory checks, 4 expected exhaustion
checks, 2 deliberate guard processes, 458 crossings in exact expected sequence, and 7 sanitizer-clean
non-guard cases**. The 24 unique requests include the guard request; deliberate guard processes are not
sanitizer-clean transport. Outer-limit stops at about 0.11551008 mm with one substep and status 2;
record-limit stops at 1 mm with status 3. World exit occurs at 10 mm with status 1.

| Case family | Max host position residual (mm) | Max device position residual (mm) | Fixed budget (mm) |
|---|---:|---:|---:|
| Zero | 5.33e-15 | 5.33e-15 | 0.005 |
| Constant / repeated crossings | 1.991e-5 | 1.970e-5 | 0.005 |
| Near-tangent hit/miss | 8.994e-7 | 8.288e-7 | 0.005 |
| Affine, including neutral control | 1.151e-6 | 1.259e-4 | 0.02 |

Maximum crossing-path residual is 2.058e-5 mm; maximum normal face residual is 1.068e-12 mm. These bounds
apply only to these synthetic cases, not a general convergence claim. Public momentum is fixed by design;
material interactions, eIon, ancestry, real composite fields/world coverage and speed are not validated.
**101 Python tests + 36 subtests pass**, including retained native navigation replay with no skips.
The immutable deployed h7a README remains the build-time record; this working-tree README records progress.

## Historical user build on UVA (already completed; do not repeat)

Prepared source: `/home/ram2aq/mu2e/fastsim_uva/celer_navigation_h7a/src`.
Build destination: `/scratch/ram2aq/mu2efastsim/celer_navigation_h7a-build`.
The user runs the whole block below from a laptop terminal on the UVA network/VPN. It allocates an A100,
checks the source/API pins, and builds **only this executable**. The 30-minute allocation is a ceiling, not a
runtime prediction. `mkdir` refuses an existing build directory: after interruption inspect the job and logs
before retrying; do not remove old outputs or blindly launch a second build.

```bash
ssh -T uva-hpc \
  'srun -A ldmxuva -p gpu --gres=gpu:a100:1 -c 4 --mem=16G -t 00:30:00 --job-name=celer-nav-h7a-build bash -l -s' <<'UVA_COMMANDS'
module load apptainer || exit 1
set -euo pipefail
ROOT=/home/ram2aq/mu2e/fastsim_uva/celer_navigation_h7a
BUILD=/scratch/ram2aq/mu2efastsim/celer_navigation_h7a-build
H5=/scratch/ram2aq/mu2efastsim/celer_rdc_h5
IMAGE=/scratch/ram2aq/mu2efastsim/mu2e-celeritas.sif
(cd "$ROOT" && sha256sum --check snapshot.sha256)
(cd "$ROOT/src" && sha256sum --check source.sha256)
(cd "$H5/install/include" && sha256sum --check "$ROOT/src/api-pins.sha256")
mkdir "$BUILD"
printf 'job=%s host=%s utc=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" | tee "$BUILD/allocation.txt"
apptainer exec --cleanenv --nv \
  --bind "$ROOT:$ROOT:ro" --bind "$BUILD:$BUILD" --bind "$H5:$H5:ro" \
  "$IMAGE" bash -s -- "$ROOT/src" "$BUILD" "$H5/install" <<'CONTAINER_COMMANDS'
set -euo pipefail
SRC=$1; BUILD=$2; PREFIX=$3
export LD_LIBRARY_PATH="$PREFIX/lib64:$PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:/opt/view:/usr/local/cuda"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCeleritas_DIR="$PREFIX/lib64/cmake/Celeritas" \
  -DM2FS_CARTMAP_DIR="$SRC" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_BUILD_RPATH="$PREFIX/lib64" \
  2>&1 | tee "$BUILD/configure.log"
cmake --build "$BUILD" --parallel 4 --verbose 2>&1 | tee "$BUILD/build.log"
ldd "$BUILD/m2fs-navigation-probe" | tee "$BUILD/ldd.txt"
sha256sum "$BUILD/m2fs-navigation-probe" | tee "$BUILD/executable.sha256"
CONTAINER_COMMANDS
UVA_COMMANDS
```

**Stop after the build and report completion.** The agent will verify source/cache bindings, actual CUDA
`-rdc=true` and device-link command, binary identity and loaded h5 libraries before any transport. Python
contract tests and a source review are not a C++ compilation/link/runtime result. Geometry/field payload and
helper lifetime errors can still require a corrected snapshot and user rebuild.

After verification, `run_probe.sh` is a separate, runtime-only batch wrapper under the snapshot root. It uses
fresh `probe/celer_navigation_h7a_<job>/` output and runs non-guard cases under Compute Sanitizer. Guard cases
use separate host/device processes without sanitizer, require the exact guard marker and no result file, and
are not counted as clean transport. The immutable original snapshot and runtime outputs must be preserved.

The scalar/control configuration generation is diagnostic, not per-event detector processing. `fixtures.py`
checks analytic helices and independently converged DOP853 solutions, all crossing records, completion/status,
geometry labels, field-domain safety, path/time accounting and fixed numerical budgets. Passing a process
exit/marker check alone is not a passing geometry result. Repeated surface errors and float/texture floors
must be reported without fitting tolerances or claiming generic monotonic convergence.
