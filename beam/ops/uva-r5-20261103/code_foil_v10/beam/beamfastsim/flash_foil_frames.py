"""Unique foil identities and sided coordinate frames; not a navigator.

Geometry is an axis-aligned annulus, in the geometry's global length units.
Foil IDs are original array indices, NOT thickness/material-type indices.
Exact surfaces are admitted, but deciding material ownership there is the
caller's job. No fuzzy membership, clipping, ray projection, direction
normalization, or finite-foil transport certificate is supplied here.

Basis columns are (e1, e2, normal) = diag(1, sign(u_z), sign(u_z)).
The anchor has the actual point's global x/y and the near z-face's z.
Thus even a rim/hole entry has local x=(0, 0, actual_depth), not zero.
After scattering the caller must retain this frame, not rebuild it from u.
Transforms incur floating-point rounding; only retained global_x is an exact
copy of the input coordinates. Transforming back is NOT a bitwise guarantee.
NumPy is the only import dependency; *_jax functions import JAX on use.
"""

from dataclasses import dataclass
import numpy as np

__all__ = ["FoilTable", "build_table", "gather", "gather_jax", "entry_frame",
           "entry_frame_jax", "validate_basis", "to_local", "to_global",
           "to_local_jax", "to_global_jax"]


def _immutable(a):
    a = np.ascontiguousarray(a)
    # An immutable bytes owner prevents even setflags(write=True).
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class FoilTable:
    """Host-validated table; construct using build_table, not this constructor.

    All columns are immutable host arrays. material_type indexes the separate
    sorted material_thickness (= unique 2*ht) table; it never identifies a foil.
    """
    axis_x: float
    foil_id: np.ndarray
    foil_z: np.ndarray
    foil_rin: np.ndarray
    foil_rout: np.ndarray
    foil_ht: np.ndarray
    material_type: np.ndarray
    material_thickness: np.ndarray


def build_table(geom):
    """Copy/validate BeamGeometry (or an object with .beam) on the host.

    Input order defines identity and is preserved, including equal thicknesses.
    Require finite nonempty 1-D columns, valid radii/positive ht, and disjoint
    closed z intervals (no duplicates/touching foils). No geometry files are read.
    """
    geom = getattr(geom, "beam", geom)
    raw = [np.asarray(getattr(geom, n)) for n in
           ("foil_z", "foil_rin", "foil_rout", "foil_ht")]
    axis = np.asarray(geom.axis_x)
    if axis.ndim != 0 or axis.dtype.kind not in "iuf" or not np.isfinite(axis):
        raise ValueError("axis_x must be a finite real scalar")
    if any(a.dtype.kind not in "iuf" or a.ndim != 1 or not a.size
           or not np.all(np.isfinite(a)) for a in raw):
        raise ValueError("foil arrays must be nonempty finite real 1-D arrays")
    if any(a.shape != raw[0].shape for a in raw):
        raise ValueError("foil arrays must have equal lengths")
    z, ri, ro, ht = [a.astype(np.float64) for a in raw]
    if np.any(ri < 0) or np.any(ro <= ri) or np.any(ht <= 0):
        raise ValueError("require 0 <= rin < rout and ht > 0")
    with np.errstate(over="ignore", invalid="ignore"):
        lo, hi, thickness = z - ht, z + ht, 2 * ht
    if not all(np.all(np.isfinite(a)) for a in (lo, hi, thickness)) or np.any(lo >= hi):
        raise ValueError("foil faces/thickness must be finite and distinct")
    order = np.argsort(z)
    if np.any(lo[order][1:] <= hi[order][:-1]):
        raise ValueError("foil z intervals must be disjoint, without duplicates/touching")
    types, inverse = np.unique(thickness, return_inverse=True)
    return FoilTable(float(axis), *map(_immutable, (
        np.arange(z.size, dtype=np.int64), z, ri, ro, ht,
        inverse.astype(np.int64), types)))


def _ids(xp, foil_id):
    ids = xp.asarray(foil_id)
    if ids.dtype.kind not in "iu":
        raise ValueError("foil_id must have integer dtype (not bool or float)")
    return ids


def _gather(xp, table, foil_id):
    ids = _ids(xp, foil_id)
    n = table.foil_id.size
    valid = (ids >= 0) & (ids < n)
    # Append a real sentinel row: no invalid ID can alias the last foil.
    safe = xp.where(valid, ids, n).astype(xp.int32)
    result = {}
    for name in ("foil_z", "foil_rin", "foil_rout", "foil_ht", "material_type"):
        a = xp.asarray(getattr(table, name))
        sentinel = -1 if name == "material_type" else xp.nan
        result[name] = xp.concatenate((a, xp.full((1,), sentinel, a.dtype)))[safe]
    result.update(foil_id=xp.where(valid, safe, -1), valid=valid, unresolved=~valid,
                  axis_x=xp.broadcast_to(xp.asarray(table.axis_x), ids.shape))
    return result


def gather(table, foil_id):
    """Vectorized explicit-ID gather; repeats/order retained; invalid row is NaN/-1."""
    return _gather(np, table, foil_id)


def gather_jax(table, foil_id):
    """Pure JAX gather, usable in jit with a closed-over host table."""
    import jax.numpy as jnp
    return _gather(jnp, table, foil_id)


def _vector(xp, value):
    a = xp.asarray(value)
    if a.ndim < 1 or a.shape[-1] != 3 or a.dtype.kind not in "iuf":
        raise ValueError("vectors must be real arrays with final dimension 3")
    return a


def _basis_ok(xp, basis):
    gram = xp.swapaxes(basis, -1, -2) @ basis
    # Explicit 3x3 determinant avoids a linalg dependency on the JAX lane path.
    a, b, c = basis[..., 0, :], basis[..., 1, :], basis[..., 2, :]
    det = (a[..., 0] * (b[..., 1]*c[..., 2] - b[..., 2]*c[..., 1])
           - a[..., 1] * (b[..., 0]*c[..., 2] - b[..., 2]*c[..., 0])
           + a[..., 2] * (b[..., 0]*c[..., 1] - b[..., 1]*c[..., 0]))
    eps = xp.finfo(basis.dtype).eps
    return (xp.all(xp.isfinite(basis), axis=(-2, -1))
            & xp.all(xp.abs(gram - xp.eye(3, dtype=basis.dtype)) <= 64 * eps, axis=(-2, -1))
            & (xp.abs(det - 1) <= 64 * eps))


def _entry(xp, table, foil_id, x, u, B):
    x, u, B = (_vector(xp, v) for v in (x, u, B))
    dtype = xp.result_type(x.dtype, u.dtype, B.dtype, xp.float32)
    if dtype not in (xp.dtype("float32"), xp.dtype("float64")):
        raise ValueError("only float32/float64 frames supported")
    ids = _ids(xp, foil_id)
    shape = xp.broadcast_shapes(ids.shape, x.shape[:-1], u.shape[:-1], B.shape[:-1])
    ids = xp.broadcast_to(ids, shape)
    # Keep originals separately: no normalization or dtype conversion of global_x.
    gx, gu, gb = (xp.broadcast_to(v, shape + (3,)) for v in (x, u, B))
    xx, uu, bb = (v.astype(dtype) for v in (gx, gu, gb))
    row = _gather(xp, table, ids)
    z, ri, ro, ht, ax = (row[k].astype(dtype) for k in
                         ("foil_z", "foil_rin", "foil_rout", "foil_ht", "axis_x"))
    sg = xp.sign(uu[..., 2])
    diag = xp.stack((xp.ones_like(sg), sg, sg), axis=-1)
    basis = diag[..., :, None] * xp.eye(3, dtype=dtype)
    face = z - sg * ht
    anchor = xp.stack((xx[..., 0], xx[..., 1], face), axis=-1)
    depth = sg * (xx[..., 2] - face)
    radius = xp.hypot(xx[..., 0] - ax, xx[..., 1])
    lo, hi = z - ht, z + ht
    inside = ((radius >= ri) & (radius <= ro)
              & (xx[..., 2] >= lo) & (xx[..., 2] <= hi))
    interior = ((radius > ri) | (ri == 0)) & (radius < ro) & (xx[..., 2] > lo) & (xx[..., 2] < hi)
    finite = xp.all(xp.isfinite(xx) & xp.isfinite(uu) & xp.isfinite(bb), axis=-1)
    eps = xp.finfo(dtype).eps
    unit = xp.abs(xp.sum(uu * uu, axis=-1) - 1) <= 64 * eps
    valid = (row["valid"] & finite & unit & (sg != 0) & inside
             & (depth >= 0) & (depth <= 2 * ht) & _basis_ok(xp, basis)
             & (ri >= 0) & (ro > ri) & (ht > 0) & (lo < hi)
             & xp.isfinite(ax) & xp.isfinite(2 * ht))
    def masked(v, rank=0):
        return xp.where(valid[(...,) + (None,) * rank], v, xp.asarray(xp.nan, dtype))
    local_x = xp.stack((xp.zeros_like(depth), xp.zeros_like(depth), depth), axis=-1)
    return dict(basis=masked(basis, 2), anchor=masked(anchor, 1),
                local_x=masked(local_x, 1), local_u=masked(diag * uu, 1),
                local_B=masked(diag * bb, 1), depth=masked(depth),
                global_x=gx, global_u=gu, global_B=gb,
                foil_id=row["foil_id"], material_type=row["material_type"],
                valid=valid, unresolved=~valid, interior=valid & interior,
                boundary=valid & ~interior)


def entry_frame(table, foil_id, x, u, B):
    """Build sided frames; malformed numeric/nonunit/zero-u_z/outside lanes unresolved.

    Outputs are broadcast arrays. Unresolved local data/basis/anchor/depth are
    NaN, but global inputs remain untouched; inspect valid before using them.
    No tolerance admits a point outside the annulus. Numerical collapse or a
    depth outside [0,2ht] is refused rather than silently repaired.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        return _entry(np, table, foil_id, x, u, B)


def entry_frame_jax(table, foil_id, x, u, B):
    """Pure JAX lane helper, with the same masks and no host numeric validation."""
    import jax.numpy as jnp
    return _entry(jnp, table, foil_id, x, u, B)


def validate_basis(basis):
    """Host validation of right-handed orthonormal basis columns; raises ValueError."""
    b = np.asarray(basis)
    if b.shape[-2:] != (3, 3) or b.dtype.kind != "f" or not np.all(_basis_ok(np, b)):
        raise ValueError("basis must be a finite orthonormal proper rotation")


def _transform(xp, frame, value, point, inverse):
    v = _vector(xp, value)
    b, a = xp.asarray(frame["basis"]), xp.asarray(frame["anchor"])
    if b.shape[-2:] != (3, 3):
        raise ValueError("basis must have final shape (3,3)")
    if inverse:
        out = xp.einsum("...ij,...j->...i", b, v)
        if point:
            out = out + a
    else:
        out = xp.einsum("...ji,...j->...i", b, v - a if point else v)
    ok = xp.asarray(frame["valid"]) & _basis_ok(xp, b)
    return xp.where(ok[..., None], out, xp.nan)


def to_local(frame, value, *, point=False):
    """Rotate a global vector; point=True also subtracts anchor. Invalid => NaN.

    Uses the saved basis even if a scattered direction now points backwards.
    """
    return _transform(np, frame, value, point, False)


def to_global(frame, value, *, point=False):
    """Rotate a local vector; point=True adds anchor. Roundtrip is approximate."""
    return _transform(np, frame, value, point, True)


def to_local_jax(frame, value, *, point=False):
    """Pure JAX equivalent of to_local (point is a static Python flag)."""
    import jax.numpy as jnp
    return _transform(jnp, frame, value, point, False)


def to_global_jax(frame, value, *, point=False):
    """Pure JAX equivalent of to_global (point is a static Python flag)."""
    import jax.numpy as jnp
    return _transform(jnp, frame, value, point, True)
