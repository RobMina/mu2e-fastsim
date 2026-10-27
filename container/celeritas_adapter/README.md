# Celeritas 0.6.3 Cartesian-field and rootless-event adapter

**October 27 follow-up:** the existing user-built h5 adapter and h6 probe have been exercised; see
[control sensitivity, thin-surface results and remaining Mu2e gates](n5-results.md). The preparation/build
sections below describe the original h4 handoff, not a request to rebuild. Use the repaired h5 library prefix
for runtime; the original image's Cartesian device kernel must not be reused unpatched.

**h7 navigation source preparation:** [new probe and user build instructions](navigation_probe/README.md)
add direct host/device geometry crossing records without deposit filtering. This is a separate target linking
h5; it is not yet compiled or validated and does not change the existing adapter.

This is a bounded application extension, not a new shower model or a Mu2e flash evaluator. It prepares a patched
copy of the **v0.6.3 celer-g4 application** and links it against the existing image's installed libraries. The
image and its installed libraries are not modified. Compilation and executable tests must be run by the user.
Read `../celeritas-feasibility.md` for the completed stock-image smoke test and remaining physics gates.

## Contract

- `field_type: "cartmap"`, `field_file: "map.json"` selects the same Celeritas Cartesian input for the host
  Geant4 field and the GPU along-step factory. The existing uniform/RZ paths remain available.
- Map format `m2fs-cartmap-v1` has explicit **mm**, **tesla**, `gdml-world` coordinates and `[x][y][z][component]`
  (C-order) storage. Axes have `min_mm`, `max_mm`, integer `num`; JSON field values are flat. Conversion uses
  Celeritas/CLHEP unit helpers, not an assumed native build unit system.
- **Only world-contained maps are supported.** The GDML world's bounding box must be inside the grid by at least
  one grid cell on every face. A rotated world is rejected. Both arms then remain within interpolation support.
  This restriction deliberately prevents running a DS-only map over a full Mu2e world without an explicit,
  validated outside-map policy. It is not a zero/clamp substitution for Offline's field selection.
- v0.6.3's host covfie path is a float linear interpolator; its CUDA path uses texture interpolation. This is not
  identical to double-precision Offline/JAX interpolation. The v0.7 clamping claim must not be assumed for v0.6.3.
  The one-cell guard is not a proof of numerical propagator behavior: exercise boundary/step overshoot in tests.
- `M2FS_EVENT_OUTPUT=/new/path/events.jsonl` enables a rootless diagnostic writer with `sd_type: "event_hit"`.
  Use serial Geant4 (or one worker). It rejects kill-offload/nonfatal-flush modes and refuses existing files, multiple workers, aborted events, absent sensitive
  collections, foreign hit types and nonfinite/negative energy. After each offload flush it writes an event row,
  including zero-hit events, generated-primary count, per-collection total edep and the existing hit attributes.
  It fails on write/flush errors. It is **total deposited energy**, not StrawGasStep ionising energy; volume instance
  and leaf copy number are not a full Mu2e straw identifier. No ancestry or stock time/aggregation cuts are supplied.
- The installed `HitProcessor::num_hits_` diagnostic-counter bug remains in the library; this application never
  uses that counter. Event JSONL hit counts come from the actual Geant4 collections.

## Prepare (no build)

`prepare.py UPSTREAM_SOURCE NEW_OUTPUT` accepts only the pinned application file hashes from Celeritas commit
`ddf6c690e37d53b56d001e6bf0c26cb1b14963fa` (v0.6.3). It validates every input before writing and never edits the
upstream tree. Outputs include a provenance manifest, license notices, patched app sources and this adapter.
The fresh output is technical source only; no Mu2e inputs enter it. No configure/compiler command is run by it.

Generate a public constant/linear synthetic map with `field_contract.py synthetic NEW_MAP.json`; use
`field_contract.py check MAP.json` for schema checks. The synthetic map covers the existing smoke world. It is
not a converted DSMap. A real map adapter must additionally establish the Mu2e-to-GDML transform and complete
field-domain coverage from the sample's resolved geometry/configuration, not copy the beam stage's arrays blindly.

## User build (inside the existing image, on an allocated UVA node)

Prepared source is deployed at `/home/ram2aq/mu2e/fastsim_uva/celer_adapter_h4` (h2/h3 are superseded preparation
snapshots; do not use them). Its manifest SHA256 is
`774e93dc8e2f8aec55b0e76947cb237c72300d5e1d122ccc89ca08b57fe28933`, verified after mirror sync. All **29 Python
contract/preparation tests pass**, including the exact upstream hashes. No compiler/configure command was run.
These are manual user commands, not a build script. From `uva-hpc`, obtain a normal short queue allocation:

```bash
srun -A ldmxuva -p gpu --gres=gpu:a100:1 -c 2 --mem=8G -t 00:20:00 --pty bash -l
module load apptainer
SRC=/home/ram2aq/mu2e/fastsim_uva/celer_adapter_h4
BUILD=/scratch/ram2aq/mu2efastsim/celer_adapter_h4-build
mkdir "$BUILD"
IMAGE=/scratch/ram2aq/mu2efastsim/mu2e-celeritas.sif
apptainer exec --cleanenv --nv --bind "$SRC:$SRC:ro" --bind "$BUILD:$BUILD" "$IMAGE" \
  cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
apptainer exec --cleanenv --nv --bind "$SRC:$SRC:ro" --bind "$BUILD:$BUILD" "$IMAGE" \
  cmake --build "$BUILD" -j 2
```

No full Geant4/Celeritas/VecGeom rebuild is intended. The external app CMake must still be tested against the image's
exported CUDA-RDC dependencies; preparation/unit tests cannot certify linking. Tests after a successful user build:

1. `m2fs-field-check MAP.json POINTS.json` evaluates the actual host map in mm/T; compare constant and asymmetric
   affine maps to analytic values. Include cell interiors and nodes; queries require a one-cell margin too,
   avoiding float rounding onto an unsupported upper endpoint. Test rejection outside that interior domain.
2. Run existing synthetic G4/GPU uniform controls with `m2fs-celer-g4`, then replace the constant uniform field with
   an identical Cartesian map. Use independently tagged outputs and reuse the saved stock uniform baselines.
3. Enable event JSONL for e-/e+/gamma; check exactly the finished event count, empty events, primary count, sums of
   hit/collection/run energy and agreement with `simple_calo` controls. Test missing SD and output-file failures.
4. Check an affine/nonuniform map, host versus GPU transport, step convergence and guard rejection before any DSMap.

**Status:** preparation and Python contract tests only; C++ not compiled, no adapter transport run. No claim of
Mu2e-compatible field behavior, production event identity, objective closure, or a commissioned `flash_point`.
