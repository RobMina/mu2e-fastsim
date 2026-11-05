"""Convex bounds/coverage tests, not an ODE or native certification."""

import numpy as np
import pytest

from beamfastsim import tsbound as b, tsfield
from test_tsfield import write_map


def test_node_bound_encloses_batched_trilinear_values(tmp_path):
    item = tsfield.load(write_map(tmp_path))
    bound = b.magnitude(item, chunk_nodes=7)
    points = np.random.default_rng(41).uniform(item.lo, item.hi, (10000, 3))
    assert np.all(np.hypot.reduce(item.field(points), axis=1) <= bound.magnitude_tesla + 1e-14)
    assert bound.node_count == int(np.prod(item.n))
    assert not bound.floating_point_certified and not bound.native_query_certified
    assert b.covered_box(item, item.lo, item.hi) == bound


def test_reflection_zero_plane_and_gap_are_not_confused(tmp_path):
    item = tsfield.load(write_map(tmp_path, mirror=True, constant=[3., 4., 0]))
    assert b.covered_box(item, [0, -1, 0], [1, 1, 1]).magnitude_tesla == 5
    other = tsfield.load(write_map(tmp_path, name='gap', lo=[0, 1, 0], mirror=True))
    with pytest.raises(ValueError, match='wholly covered'):
        b.covered_box(other, [0, -1, 0], [1, 1, 1])
    assert b.covered_box(other, [0, -2, 0], [1, -1, 1]).node_count > 0


def test_invalid_inputs(tmp_path):
    item = tsfield.load(write_map(tmp_path))
    for lo, hi in (([0, 0], [1, 1]), ([1, 0, 0], [0, 1, 1]),
                   ([-1, 0, 0], [1, 1, 1]), ([np.nan, 0, 0], [1, 1, 1])):
        with pytest.raises(ValueError):
            b.covered_box(item, lo, hi)
    for chunk in (0, True, 1.5, 1000001):
        with pytest.raises(ValueError):
            b.magnitude(item, chunk_nodes=chunk)
