"""Additional conservative 1-Lipschitz clearances, positive inside, not a parser.

These synthetic geometry primitives carry the floating-point and child-contract
limitations of tsnav; they provide neither native geometry equivalence nor transport.
"""

from dataclasses import dataclass

import numpy as np

from . import tsnav


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@dataclass(frozen=True)
class Cone(tsnav._Solid):
    """Radii 1/2 at z=-h/+h; equality only at one zero-radius boundary apex.

    Two zero inner radii remove the inner constraint; the axis stays interior.
    """

    rmax1: float
    rmax2: float
    half_length: float
    rmin1: float = 0
    rmin2: float = 0

    def __post_init__(self):
        for name in ('rmax1', 'rmax2', 'half_length', 'rmin1', 'rmin2'):
            value = tsnav._scalar(getattr(self, name), name,
                                  positive=name == 'half_length',
                                  nonnegative=name != 'half_length')
            object.__setattr__(self, name, value)
        for outer, inner in ((self.rmax1, self.rmin1), (self.rmax2, self.rmin2)):
            if inner > outer or (inner == outer and outer != 0):
                raise ValueError('require inner < outer except at a zero apex')
        if self.rmax1 == self.rmax2 == 0:
            raise ValueError('cone must have nonzero volume')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                for label, a, b in (('outer', self.rmax1, self.rmax2),
                                    ('inner', self.rmin1, self.rmin2)):
                    slope = ((np.float64(b) - a) / 2) / self.half_length
                    slope = tsnav._scalar(slope, 'radius slope')
                    if a != b and slope == 0:
                        raise ValueError('nonzero conical slope underflows')
                    norm = np.hypot(1, slope)
                    object.__setattr__(self, '_' + label, (a / 2 + b / 2, slope, norm))
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError('nonfinite conical slope') from exc

    def _clearance(self, p):
        radial = np.hypot(p[:, 0], p[:, 1])
        middle, slope, norm = self._outer
        result = np.minimum((middle - radial) / norm + (slope / norm) * p[:, 2],
                            self.half_length - np.abs(p[:, 2]))
        if self.rmin1 != 0 or self.rmin2 != 0:
            middle, slope, norm = self._inner
            result = np.minimum(result, (radial - middle) / norm -
                                (slope / norm) * p[:, 2])
        return result


@dataclass(frozen=True, eq=False)
class Sector(tsnav._Solid):
    """Counterclockwise azimuthal cut in radians; the axis is a cut boundary."""

    solid: object
    startphi: float
    deltaphi: float

    def __post_init__(self):
        tsnav._child(self.solid)
        start = tsnav._scalar(self.startphi, 'startphi') % (2 * np.pi)
        delta = tsnav._scalar(self.deltaphi, 'deltaphi', positive=True)
        if delta > 2 * np.pi:
            raise ValueError('deltaphi must not exceed 2*pi')
        end = (start + delta) % (2 * np.pi)
        normals = [[-np.sin(start), np.cos(start)], [np.sin(end), -np.cos(end)]]
        object.__setattr__(self, 'startphi', start)
        object.__setattr__(self, 'deltaphi', delta)
        object.__setattr__(self, '_normals', tsnav._fixed(normals, (2, 2), 'normals'))

    def _clearance(self, p):
        result = tsnav._query(self.solid, p)
        if self.deltaphi == 2 * np.pi:
            return result
        sides = p[:, :2] @ self._normals.T
        angular = (np.min(sides, axis=1) if self.deltaphi <= np.pi
                   else np.max(sides, axis=1))
        return np.minimum(result, angular)


@dataclass(frozen=True, eq=False)
class Extrusion(tsnav._Solid):
    """Simple constant polygon prism, either winding; no internal triangulation seams."""

    vertices: np.ndarray
    half_length: float

    def __post_init__(self):
        vertices = tsnav._real(self.vertices, 'vertices')
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError('vertices must have shape (M,2), M >= 3')
        vertices = tsnav._fixed(vertices, vertices.shape, 'vertices')
        height = tsnav._scalar(self.half_length, 'half_length', positive=True)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                edges = np.roll(vertices, -1, axis=0) - vertices
                lengths = np.hypot(edges[:, 0], edges[:, 1])
                if np.any(lengths == 0) or len(np.unique(vertices, axis=0)) != len(vertices):
                    raise ValueError('zero edges or duplicate vertices')
                following = np.roll(edges, -1, axis=0)
                if np.any((_cross(edges, following) == 0) &
                          (np.sum(edges * following, axis=1) < 0)):
                    raise ValueError('collinear backtracking')
                shifted = vertices - vertices[0]
                if np.sum(_cross(shifted, np.roll(shifted, -1, axis=0))) == 0:
                    raise ValueError('degenerate polygon')
                self._validate_simple(vertices, edges)
                units = edges / lengths[:, None]
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError('nonfinite derived polygon geometry') from exc
        object.__setattr__(self, 'vertices', vertices)
        object.__setattr__(self, 'half_length', height)
        for name, data in (('_edges', edges), ('_lengths', lengths), ('_units', units)):
            object.__setattr__(self, name, tsnav._fixed(data, data.shape, name))

    @staticmethod
    def _validate_simple(vertices, edges):
        n = len(vertices)
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                a, b = vertices[i], vertices[(i + 1) % n]
                c, d = vertices[j], vertices[(j + 1) % n]
                overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
                                 np.minimum(np.maximum(a, b), np.maximum(c, d)))
                s1, s2 = np.sign(_cross(edges[i], c - a)), np.sign(_cross(edges[i], d - a))
                t1, t2 = np.sign(_cross(edges[j], a - c)), np.sign(_cross(edges[j], b - c))
                if overlap and s1 * s2 <= 0 and t1 * t2 <= 0:
                    raise ValueError('polygon must be simple without self-touching')

    def _clearance(self, p):
        xy = p[:, :2]
        distance = np.full(len(p), np.inf)
        inside = np.zeros(len(p), dtype=bool)
        for a, edge, length, unit in zip(self.vertices, self._edges,
                                        self._lengths, self._units):
            relative = xy - a
            along = np.clip(relative @ unit, 0, length)
            residual = relative - along[:, None] * unit
            distance = np.minimum(distance, np.hypot(residual[:, 0], residual[:, 1]))
            crosses = (a[1] > xy[:, 1]) != (a[1] + edge[1] > xy[:, 1])
            inside ^= crosses & ((_cross(edge, relative) > 0) == (edge[1] > 0))
        signed = np.where(inside, distance, -distance)
        return np.minimum(signed, self.half_length - np.abs(p[:, 2]))
