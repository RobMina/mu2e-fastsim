"""CRV bar crossings: the generic slab method of mu2efastsim.crossings applied to the CRV bar boxes.

V0 found every bar axis-aligned and none rotated, so a bar is the box center +- halflen (geometry.py).
"""
from mu2efastsim import crossings as _core
from mu2efastsim.crossings import HAVE_JAX, slab_crossings  # noqa: F401  (re-exported for the CRV code and tests)


def bar_bounds(geom):
    return geom.center - geom.halflen, geom.center + geom.halflen


def crossings(origin, direction, geom, chunk: int = 32768, backend=None):
    """Sparse crossings with the CRV bars: rows (track index, bar index, t_in, t_out, path length)."""
    lo, hi = bar_bounds(geom)
    return _core.crossings(origin, direction, lo, hi, chunk=chunk, backend=backend)
