"""Small NumPy CSG clearance primitives, positive inside and zero on boundaries.

Min/max composition gives a conservative 1-Lipschitz clearance in exact
arithmetic, not an exact CSG distance. HalfSpace means normal @ point <= offset.
Placed uses an active local-to-global rotation: local = (global-origin) @ rotation.
Children may expose clearance(points) or be callable; callers must ensure their
positive-inside, 1-Lipschitz semantics. Child results are checked on every query.
"""

from dataclasses import dataclass

import numpy as np

LIMITATIONS = (
    "Floating arithmetic and rotation tolerances are not rigorous error bounds. "
    "The caller must provide complete geometry; no nominal geometry is supplied. "
    "No native geometry equivalence, hierarchy, materials or full transport is implied. "
    "Arbitrary children must themselves satisfy the clearance contract."
)


def _real(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be real numeric data")
        with np.errstate(over="raise", invalid="raise"):
            result = np.array(raw, dtype=float, copy=True)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite")
        return result
    except (TypeError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f"invalid {name}") from exc


def _scalar(value, name, positive=False, nonnegative=False):
    result = _real(value, name)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar")
    result = float(result)
    if (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"invalid {name}")
    return result


def _fixed(value, shape, name):
    result = _real(value, name)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(shape)


def _child(solid):
    method = getattr(solid, "clearance", None)
    if method is None:
        method = solid
    if not callable(method):
        raise ValueError("child must provide callable clearance or be callable")
    return method


def _result(value, n):
    result = _real(value, "clearance")
    if result.shape != (n,):
        raise ValueError("clearance must have shape (N,)")
    return result


def _query(solid, points):
    return _result(_child(solid)(points), len(points))


class _Solid:
    def clearance(self, points):
        points = _real(points, "points")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N,3)")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return _result(self._clearance(points), len(points))
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("nonfinite derived geometry result") from exc


@dataclass(frozen=True)
class Sphere(_Solid):
    radius: float

    def __post_init__(self):
        object.__setattr__(self, "radius", _scalar(self.radius, "radius", positive=True))

    def _clearance(self, p):
        return self.radius - np.hypot.reduce(p, axis=1)


@dataclass(frozen=True, eq=False)
class Box(_Solid):
    half_lengths: np.ndarray

    def __post_init__(self):
        lengths = _fixed(self.half_lengths, (3,), "half_lengths")
        if np.any(lengths <= 0):
            raise ValueError("half_lengths must be positive")
        object.__setattr__(self, "half_lengths", lengths)

    def _clearance(self, p):
        return np.min(self.half_lengths - np.abs(p), axis=1)


@dataclass(frozen=True)
class Tube(_Solid):
    rmax: float
    half_length: float
    rmin: float = 0

    def __post_init__(self):
        for name in ("rmax", "half_length", "rmin"):
            value = _scalar(getattr(self, name), name, positive=name != "rmin",
                            nonnegative=name == "rmin")
            object.__setattr__(self, name, value)
        if self.rmin >= self.rmax:
            raise ValueError("require rmin < rmax")

    def _clearance(self, p):
        radial = np.hypot(p[:, 0], p[:, 1])
        result = np.minimum(self.rmax - radial, self.half_length - np.abs(p[:, 2]))
        return np.minimum(result, radial - self.rmin) if self.rmin > 0 else result


@dataclass(frozen=True)
class Torus(_Solid):
    """Full torus about z, with major_radius > minor_radius > 0."""

    major_radius: float
    minor_radius: float

    def __post_init__(self):
        for name in ("major_radius", "minor_radius"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, positive=True))
        if self.major_radius <= self.minor_radius:
            raise ValueError("require major_radius > minor_radius")

    def _clearance(self, p):
        radial = np.hypot(p[:, 0], p[:, 1])
        return self.minor_radius - np.hypot(radial - self.major_radius, p[:, 2])


@dataclass(frozen=True, eq=False)
class HalfSpace(_Solid):
    normal: np.ndarray
    offset: float

    def __post_init__(self):
        normal = _fixed(self.normal, (3,), "normal")
        offset = _scalar(self.offset, "offset")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                length = np.hypot.reduce(normal)
                if length == 0:
                    raise ValueError("normal must be nonzero")
                normal = _fixed(normal / length, (3,), "normal")
                offset = _scalar(offset / float(length), "offset")
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("nonfinite normalized half-space") from exc
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", offset)

    def _clearance(self, p):
        return self.offset - p @ self.normal


@dataclass(frozen=True, init=False)
class _Composite(_Solid):
    solids: tuple

    def __init__(self, *solids):
        if not solids:
            raise ValueError("composite requires at least one child")
        for solid in solids:
            _child(solid)
        object.__setattr__(self, "solids", tuple(solids))


@dataclass(frozen=True, init=False)
class Intersection(_Composite):
    def _clearance(self, p):
        return np.minimum.reduce([_query(s, p) for s in self.solids])


@dataclass(frozen=True, init=False)
class Union(_Composite):
    def _clearance(self, p):
        return np.maximum.reduce([_query(s, p) for s in self.solids])


@dataclass(frozen=True)
class SplitUnion(_Solid):
    """Union of A clipped by H and B clipped by the algebraic opposite of H.

    A and B are UNCLIPPED remainders. The redundant A-intersect-B term removes
    the shared-plane zero only where both are strictly interior. No fitted
    matching of independent faces, gap closure or universal seam removal occurs.
    """

    first: object
    second: object
    plane: HalfSpace

    def __post_init__(self):
        _child(self.first)
        _child(self.second)
        if type(self.plane) is not HalfSpace:
            raise ValueError("SplitUnion requires one explicit HalfSpace")

    def _clearance(self, p):
        a, b, h = _query(self.first, p), _query(self.second, p), _query(self.plane, p)
        return np.maximum.reduce((np.minimum(a, h), np.minimum(b, -h), np.minimum(a, b)))


@dataclass(frozen=True)
class Difference(_Solid):
    first: object
    second: object

    def __post_init__(self):
        _child(self.first)
        _child(self.second)

    def _clearance(self, p):
        return np.minimum(_query(self.first, p), -_query(self.second, p))


@dataclass(frozen=True, eq=False)
class Placed(_Solid):
    solid: object
    origin: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        _child(self.solid)
        origin = _fixed(self.origin, (3,), "origin")
        rotation = _fixed(self.rotation, (3, 3), "rotation")
        try:
            with np.errstate(over="raise", invalid="raise"):
                gram = rotation.T @ rotation
                determinant = np.linalg.det(rotation)
            if not (np.all(np.isfinite(gram)) and np.isfinite(determinant)
                    and np.allclose(gram, np.eye(3), rtol=0, atol=1e-12)
                    and np.isclose(determinant, 1, rtol=0, atol=1e-12)):
                raise ValueError("rotation must be proper orthogonal")
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise ValueError("invalid rotation") from exc
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "rotation", rotation)

    def _clearance(self, p):
        local = p - self.origin
        local = local @ self.rotation
        if not np.all(np.isfinite(local)):
            raise ValueError("nonfinite local coordinates")
        return _query(self.solid, local)
