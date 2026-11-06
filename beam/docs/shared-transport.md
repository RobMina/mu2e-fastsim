# Shared transport structure

Share physics and geometry mathematics; specialize detector search, execution
and response. This structure does not promote a new transport model or change
CE/flash cuts, step limits, random streams or defaults.

## Ownership and dependencies

```text
mu2efastsim/                         detector-independent core
  eslab_jax.py                      one electron/positron material-step body
    lane_step                       state only, optional boundary/safety
    lane_step_photons               same step plus optional brems record
  geometry/
    __init__.py                     lightweight namespace, no JAX import
    annulus_jax.py                  restricted finite annular queries

beamfastsim/                        Mu2e beam/target-specific code
  target_geometry_jax.py            shared material types, maps and frames
  cechain_jax.py                    CE navigation, refill and state outputs
  flash_navigator_jax.py            charge-aware flash navigation and records
  flash_hybrid.py                   sparse CPU vacuum scheduling
  flash_annulus_jax.py              compatibility re-export from core
```

Core imports no detector package. The target helper imports beam geometry,
material/cut conventions and field-map code, so it belongs in beam, not core.
Detector response and reconstruction stay in their respective packages.

## Shared now

Both device navigators call the same `target_geometry_jax` implementations for
material/thickness typing, field loading and map precedence, momentum/beta,
fixed-shape gather, geometry arrays, radial/OPA coordinates, layer frames and
bracketed shell bisection. Existing helper names on each navigator are aliases.
Foil radii and faces retain actual foil indexing, distinct from material type.
The electron and positron shell-curvature signs are selected statically.

`lane_step` and `lane_step_photons` accept the same paired keyword-only
`boundary=None, safety=None` callbacks. Both use one underlying core step. The
state-only path does not allocate a brems record, so finite-boundary callers do
not need photon buffers. `electron_slab` remains the slab-level interface without
these hooks. The complete callback contract is in core's `docs/photon-records.md`.

The annular geometry implementation and its main tests live in core. The beam
compatibility module contains no duplicate solver. Beam retains a cross-package
check against its independent photon-ray calculation and physical Al-table
core/annulus integration tests. The core tests need no beam installation.

## Deliberately separate

The two navigator loops, caches, candidate selection, RNG scheduling, state
recording and time integration remain separate. CE refill/compaction and flash
fixed material batches are different execution contracts; unifying them could
change individual random histories. Flash's hybrid CPU bursts are not a default
CE scheduling policy. No tracker panel/straw indexing, CRV bar indexing, calo
crystal lookup or detector readout moves into this module.

`TargetKernels` closes over geometry at setup. These are ordinary traceable
functions invoked inside each caller's existing JIT: no Python callback per
particle, per-step device transfer or runtime detector dispatch is introduced.
The structure permits specialized compiled programs; it is not a measured speed
improvement and requires separate performance checks on each adopting chain.

## Geometry limitations and the next integration

Moving the annulus solver does not widen its support. It remains a float64
straight/axial-field development primitive. Unsupported or unresolved queries
return `supported=False` with NaN distance, never a clear-path infinity. Do not
feed that NaN directly to the material-step callback. Resolve the geometry or
stop explicitly before committing transport.

Neither CE nor flash enables the new boundary hooks yet. Their charged foils
remain laterally infinite slabs; frames use axial-face origins and the shell
bisection requires an already selected crossing bracket. It cannot certify a
first curved exit or leave/return trajectory. Next steps are a general-field
boundary query, persistent foil identity, actual-depth lateral entry and a
conservative vacuum candidate search shared by normal and compact/hybrid paths.
See [flash-finite-foils.md](flash-finite-foils.md).

## Regression checks

`tests/test_target_geometry_jax.py` checks shared routing, material-versus-foil
identity, face origins, frame handedness, charge-dependent shell curvature,
map priority and gather order. Existing CE and flash suites exercise the actual
callers; this is not a new detector physics validation.

For an exact local before/after check of a structural edit, run in the same
CPU/JAX environment before editing, then compare after editing:

```bash
SNAPSHOT_DIR=$(mktemp -d)
SNAPSHOT="$SNAPSHOT_DIR/target-before.npz"
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python3 tests/target_navigation_snapshot.py --out "$SNAPSHOT"
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python3 tests/target_navigation_snapshot.py --reference "$SNAPSHOT"
```

The fixture covers CE and both flash charges, float32/64, zero and tilted fields,
foil births/entries, shells/rings, inactive lanes and partial steps. It compares
all raw arrays including slab state, random iteration, frames and photon records.
The extraction check matched 1,096 arrays byte-for-byte across 24 cases. This
statement applies to that fixture/environment, not every geometry or device.
