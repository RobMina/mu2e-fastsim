# h6 field values and ODE convergence probe

**Built by the user and exercised on A100, 2026-10-26.** Build job **21983481** and six-case memcheck job
**21983494** completed successfully. Independent replay passes after two documented Python checker fixes;
no C++/fixture/budget changes. See [measured results and limitations](runtime-results.md). This separate
`m2fs-field-probe` links the user-built h5 install without modifying h4/h5, the image or dependencies.
It is not a Mu2e field-map or flash evaluator. **Do not repeat the completed build or GPU run for replay.**

## What is exercised

The probe evaluates the actual installed Celeritas 0.6.3 `CartMapFieldParams`/`CartMapField` on the host and
CUDA device, using the h4 parser unchanged. It integrates the actual Celeritas `MagFieldEquation`,
`DormandPrinceIntegrator` and `FieldSubstepper` with explicit charge and momentum, returning endpoint/path,
momentum drift, step counts, largest step, evaluation counts and minimum query-to-guard margin. Fixed steps
and adaptive chord/error controls are tested separately; the latter accumulates returned substep lengths.
One CUDA thread handles each independent sample or trajectory. C++ host parsing and independent-test cases
are small; NumPy batches the Python oracles across trajectories. There is no per-event detector analysis.

Host evaluation is compiled in `.cc`, device evaluation in `.cu`: Celeritas selects `MemSpace::native` from
the translation unit. Device activation precedes map construction, the owner outlives all launches, and
launch/synchronization errors are fatal. Device output is zero-initialized and every record needs the kernel's
completion marker. An A100 is required; no CPU fallback can satisfy a GPU check. Explicit Celeritas/covfie
compile dependencies and the RDC-aware link wrapper are used.

**Scope limitation:** this is the field/ODE/substepper layer, not the geometry-aware `FieldPropagator`,
VecGeom boundary navigator, material/MSC transport or Geant4 hit production. The h5 regression already covers
the repaired Cartesian action kernel. This probe cannot replace that test or establish full navigation
boundary behavior. No total-edep score is used as a field-accuracy oracle.

## Domain and failure policy

Every field evaluation, including rejected Runge–Kutta trial stages, checks the same closed one-cell-inset
map domain. Finite queries outside it increment a sticky guard-violation counter and return NaN without
accessing host array/texture memory. The trajectory is then explicitly invalid (status 1), never accepted
with a zero/clamped substitute. Subsequent calls short-circuit after the first violation. Nonfinite queries
or returned fields have a separate numerical-failure counter and cause a fatal result, not an accepted
boundary rejection. Stagnation, invalid progress, outer trajectory-budget exhaustion and absent completion
markers also fail. Upstream `FieldSubstepper` does not expose internal iteration-budget exhaustion; this probe
does not patch it. An accepted finite substep is therefore not itself a convergence certificate: the independent
endpoint and step/tolerance comparisons remain necessary and cannot prove every internal trial converged.

Queries on all six inset faces, just inside and outside them, and trajectories deliberately crossing the
inset guard exercise this contract. These are **map-query safety boundaries**, not physical geometry faces.
Outside full-map interpolation is intentionally not attempted on either backend. The installed covfie
0.15.6 CUDA texture sets `cudaAddressModeClamp`, linear filtering and a `+0.5` texel-centre offset; the host
path is software linear interpolation without that clamp layer. This implementation fact is not permission
to use clamping as the experiment's outside-field policy.

## Independent references and acceptance

`oracle.py prepare NEW_DIRECTORY` writes public zero, tilted-constant and asymmetric affine field maps,
query/trajectory requests and independent expectations. It uses explicit mm, tesla, MeV/c and elementary
charge. Constant-field trajectories use a closed-form helix; affine trajectories use SciPy DOP853 with two
independent tolerance levels as a reference precision check. Celeritas's integration is not copied into the
oracle. The Lorentz coefficient in these units is 0.299792458; opposite charge/zero charge and transverse,
oblique directions distinguish sign and axis errors.

CPU covfie stores/evaluates floats; the GPU texture adds interpolation-weight quantization. Affine field
acceptance therefore uses an a priori componentwise bound based on **1/256 cell** times field change per
axis plus floating-point allowance. This is a conservative diagnostic backend contract, **not a Mu2e
field-tolerance requirement** or evidence that the texture matches an ideal double interpolator. Raw
CPU/device/oracle errors, step/tolerance dependence and coarse/refined-map differences are reported. Do not
silently tighten/loosen tests to match measured output or demand monotone error at a texture accuracy floor.
The exact predeclared thresholds live in `oracle.py` and the generated expectation sidecars.

## Preparation and user build

Canonical files are in `container/celeritas_adapter/field_probe/`. `prepare_probe.py` accepts the hash-pinned
h4 snapshot, copies its map parser plus these source files, writes SHA256 provenance, and never configures or
builds. Fixtures and source deploy through the UVA mirror; the prepared source manifest is checked before
runtime. Build paths:

- Source: `/home/ram2aq/mu2e/fastsim_uva/celer_field_h6/src`.
- Fixtures: `/home/ram2aq/mu2e/fastsim_uva/celer_field_h6/fixtures`.
- New build: `/scratch/ram2aq/mu2efastsim/celer_field_h6-build`.
- Existing Celeritas: `/scratch/ram2aq/mu2efastsim/celer_rdc_h5/install`.

Paste in a laptop terminal on the UVA network/VPN. The requested 30-minute allocation is a ceiling, not a
runtime estimate. The command builds **only the new probe** and never invokes a Celeritas/dependency build.
Do not repeat the h5 rebuild. `mkdir` refuses an existing h6 build; inspect any interrupted job before retrying.

```bash
ssh -T uva-hpc \
  'srun -A ldmxuva -p gpu --gres=gpu:a100:1 -c 4 --mem=16G -t 00:30:00 --job-name=celer-field-h6-build bash -l -s' <<'UVA_COMMANDS'
module load apptainer || exit 1
set -euo pipefail
SRC=/home/ram2aq/mu2e/fastsim_uva/celer_field_h6/src
BUILD=/scratch/ram2aq/mu2efastsim/celer_field_h6-build
H5=/scratch/ram2aq/mu2efastsim/celer_rdc_h5
IMAGE=/scratch/ram2aq/mu2efastsim/mu2e-celeritas.sif
mkdir "$BUILD"
apptainer exec --cleanenv --nv \
  --bind "$SRC:$SRC:ro" --bind "$BUILD:$BUILD" --bind "$H5:$H5:ro" \
  "$IMAGE" bash -s -- "$SRC" "$BUILD" "$H5/install" <<'CONTAINER_COMMANDS'
set -euo pipefail
SRC=$1; BUILD=$2; PREFIX=$3
export LD_LIBRARY_PATH="$PREFIX/lib64:$PREFIX/lib:${LD_LIBRARY_PATH:-}"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCeleritas_DIR="$PREFIX/lib64/cmake/Celeritas" \
  -DM2FS_CARTMAP_DIR="$SRC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_BUILD_RPATH="$PREFIX/lib64" \
  2>&1 | tee "$BUILD/configure.log"
cmake --build "$BUILD" --parallel 4 --verbose \
  2>&1 | tee "$BUILD/build.log"
ldd "$BUILD/m2fs-field-probe" | tee "$BUILD/ldd.txt"
CONTAINER_COMMANDS
UVA_COMMANDS
```

## After the user confirms completion

Verify the CMake bindings, compilation/RDC/final-device linking, binary timestamps, source manifest and all
runtime library paths. The executable contains its own device link, so its linked family need not contain
`libaccel_final.so` as the adapter does; the h5 library hashes are independently pinned. Then run the probe
under Compute Sanitizer on an allocated A100, check the actual error summary as well as exit status, and
replay the oracle locally using the project venv. Preserve outputs and report actual measured tolerances,
not only a pass flag. `run_probe.py` never imports NumPy/SciPy on UVA; the prepared fixture oracles and local
checker use the laptop venv. See its CLI arguments in source; execution instructions are not a build script.

The next gate after this layer is field-sensitive geometry transport and boundary navigation, followed by
Mu2e map-domain/coordinate transforms, ionising-edep scoring and a matched DS-start reference. No physical
outside-map policy, Mu2e acceptance tolerance or production performance claim is supplied here.
