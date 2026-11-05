"""Outward enclosures for one stored plane and normalized vector projection.

The mathematical plane uses exact binary64 origin and the mathematical unit
normal of the stored nonzero vector. Bounds cover operation rounding under the
restricted tsinterval contracts, NOT placement/GDML/native plane discrepancy.
Those discrepancies require separate uniform position/projection bounds.
"""

from . import tsinterval as iv


def signed(points, origin, normal):
    """Signed distance to the plane defined by the mathematical unit normal."""
    if type(points) is not iv.Interval or type(origin) is not iv.Interval or type(normal) is not iv.Interval:
        raise ValueError('explicit position/origin/normal intervals required')
    if points.lo.ndim < 1 or points.lo.shape[-1] != 3 or origin.lo.shape != (3,) or normal.lo.shape != (3,):
        raise ValueError('3-component points/origin/normal required')
    return iv.dot(iv.sub(points, origin), iv.normalize(normal))


def projection(directions, normal):
    """Enclose dot of mathematical unit directions and mathematical unit normal."""
    if type(directions) is not iv.Interval or type(normal) is not iv.Interval:
        raise ValueError('explicit direction/normal intervals required')
    if directions.lo.ndim < 1 or directions.lo.shape[-1] != 3 or normal.lo.shape != (3,):
        raise ValueError('3-component directions/normal required')
    return iv.dot(iv.normalize(directions), iv.normalize(normal))
