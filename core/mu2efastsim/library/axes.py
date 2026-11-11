"""Axes, binning and cell indices of a tabulated library (docs/libraries.md, "Description").

An :class:`Axis` is one key of a library: a name, a unit, a frame, its bin edges and what happens out of its domain.
A :class:`Grid` is the row-major product of axes; its flat cell index is ``(i0 * n1 + i1) * n2 + i2 ...``.

Bin convention (one for every library): bin ``k`` is ``edges[k] <= v < edges[k + 1]``, found with
``searchsorted(edges, v, side="right") - 1``. The last edge belongs to the overflow side. The bin of a NaN is
backend-dependent (NumPy sorts it last, JAX's compare-all search first): a NaN key is always reported as NONFINITE. Out-of-domain values are
reported by :func:`domain_codes` and :mod:`mu2efastsim.library.domain`, and the axis policy says what the lookup does
with them (``clip``: the end bin, declared; ``refuse``: the lookup raises; ``nan``: the output is NaN, declared).

The NumPy functions here run without JAX; the ``*_jax`` twins import JAX when called and give the same indices as
the NumPy versions for the same edges and values in the same dtype. Edges are cast to the dtype of the values, so a
float32 lookup bins against float32 edges; a value within one float32 ulp of an edge can fall in the neighbouring bin
compared with float64, which is part of what a library's float32-vs-float64 validation covers.
"""
from dataclasses import dataclass, field

import numpy as np

POLICIES = ("clip", "refuse", "nan")
SCALES = ("linear", "log", "irregular", "quantile", "categorical")
IN, BELOW, ABOVE, NONFINITE = 0, 1, 2, 3
CODE_NAMES = {IN: "in", BELOW: "below", ABOVE: "above", NONFINITE: "nonfinite"}


@dataclass(frozen=True)
class Axis:
    """One key of a library.

    ``name`` the key, as the lookup takes it; ``unit`` (``"mm"``, ``"MeV/c"``, ``"ns"``, ``"1"`` for dimensionless);
    ``frame`` the coordinate frame or definition (``"Mu2e"``, ``"tracker entry helix"``, ``"bar-local SiPM 0"``);
    ``edges`` the bin edges (strictly increasing, at least two); ``array`` the name the edges are stored under in the
    library file; ``scale`` how the edges were chosen; ``policy`` what an out-of-domain value does (POLICIES).
    """
    name: str
    unit: str
    edges: np.ndarray
    frame: str = ""
    array: str = ""
    scale: str = "irregular"
    policy: str = "clip"
    description: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        e = np.array(self.edges, dtype=np.float64, copy=True)
        if e.ndim != 1 or e.size < 2 or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0):
            raise ValueError(f"axis {self.name!r}: edges must be finite, strictly increasing, at least two")
        if not self.name or not self.unit:
            raise ValueError("an axis needs a name and a unit ('1' for dimensionless)")
        if self.policy not in POLICIES:
            raise ValueError(f"axis {self.name!r}: policy {self.policy!r} is not one of {POLICIES}")
        if self.scale not in SCALES:
            raise ValueError(f"axis {self.name!r}: scale {self.scale!r} is not one of {SCALES}")
        e.flags.writeable = False
        object.__setattr__(self, "edges", e)
        object.__setattr__(self, "array", self.array or f"{self.name}_edges")

    @property
    def nbins(self):
        return self.edges.size - 1

    @property
    def lo(self):
        return float(self.edges[0])

    @property
    def hi(self):
        return float(self.edges[-1])

    def describe(self):
        """The manifest entry of this axis (the edges themselves live in the library's arrays)."""
        d = dict(name=self.name, unit=self.unit, frame=self.frame, array=self.array, scale=self.scale,
                 policy=self.policy, nbins=self.nbins, lo=self.lo, hi=self.hi)
        if self.description:
            d["description"] = self.description
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_description(cls, d, edges):
        """Inverse of :meth:`describe`, with the edges read from the library."""
        a = cls(name=d["name"], unit=d["unit"], edges=edges, frame=d.get("frame", ""), array=d.get("array", ""),
                scale=d.get("scale", "irregular"), policy=d.get("policy", "clip"),
                description=d.get("description", ""), extra=dict(d.get("extra", {})))
        if a.nbins != int(d["nbins"]) or not np.isclose(a.lo, d["lo"], rtol=0, atol=0) \
                or not np.isclose(a.hi, d["hi"], rtol=0, atol=0):
            raise ValueError(f"axis {a.name!r}: stored edges disagree with the manifest (nbins, lo, hi)")
        return a


def locate(edges, v):
    """Bin index per value, clipped to [0, nbins - 1] (NumPy; int64). Out-of-domain values: :func:`domain_codes`."""
    edges = np.asarray(edges)
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)


def domain_codes(edges, v):
    """Per value: IN, BELOW (v < edges[0]), ABOVE (v >= edges[-1]) or NONFINITE (NumPy; int8)."""
    v = np.asarray(v)
    edges = np.asarray(edges)
    code = np.where(v < edges[0], BELOW, np.where(v >= edges[-1], ABOVE, IN))
    return np.where(np.isfinite(v), code, NONFINITE).astype(np.int8)


def locate_jax(edges, v):
    """JAX twin of :func:`locate` (searchsorted with method="compare_all": the same indices on every backend)."""
    import jax.numpy as jnp
    return jnp.clip(jnp.searchsorted(edges, v, side="right", method="compare_all") - 1, 0, edges.shape[0] - 2)


def domain_codes_jax(edges, v):
    """JAX twin of :func:`domain_codes` (int8), for counting inside a jitted lookup."""
    import jax.numpy as jnp
    code = jnp.where(v < edges[0], BELOW, jnp.where(v >= edges[-1], ABOVE, IN))
    return jnp.where(jnp.isfinite(v), code, NONFINITE).astype(jnp.int8)


@dataclass(frozen=True)
class Grid:
    """Row-major product of axes: flat cell = (i0 * n1 + i1) * n2 + i2 ..."""
    axes: tuple

    def __post_init__(self):
        axes = tuple(self.axes)
        if not axes or len({a.name for a in axes}) != len(axes):
            raise ValueError("a grid needs at least one axis and unique axis names")
        object.__setattr__(self, "axes", axes)

    @property
    def shape(self):
        return tuple(a.nbins for a in self.axes)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def names(self):
        return tuple(a.name for a in self.axes)

    def ravel(self, indices):
        """Flat cell from per-axis indices (a sequence in axis order); the arithmetic keeps the index dtype."""
        cell = indices[0]
        for idx, n in zip(indices[1:], self.shape[1:]):
            cell = cell * n + idx
        return cell

    def unravel(self, cell):
        return np.unravel_index(np.asarray(cell), self.shape)

    def cells(self, values, edges=None, xp=np):
        """Flat cell per row from ``values`` {axis name: array}; ``edges`` {axis name: edges array} overrides the axes'
        float64 edges (e.g. device copies in the compute dtype); ``xp`` is numpy or jax.numpy."""
        edges = edges or {}
        loc = locate if xp is np else locate_jax
        idx = [loc(edges.get(a.name, a.edges), values[a.name]) for a in self.axes]
        return self.ravel(idx)
