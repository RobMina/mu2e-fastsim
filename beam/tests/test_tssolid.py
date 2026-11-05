"""Synthetic shape/clearance checks, not a native geometry equivalence certificate."""

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from beamfastsim import tsnav
from beamfastsim.tssolid import Cone, Extrusion, Sector


STAIR = [[0, 0], [4, 0], [4, 1], [2, 1], [2, 3], [0, 3]]


class SolidTests(unittest.TestCase):
    def check(self, solid, points, expected):
        actual = solid.clearance(points)
        self.assertEqual(actual.shape, (len(points),))
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_cone_analytic_and_axis(self):
        cone = Cone(1, 3, 2)
        scale = np.hypot(1, 0.5)
        self.check(cone, [[0, 0, 0], [2, 0, 0], [3, 0, 0],
                          [0, 0, -2], [0, 0, 2], [0, 0, 3]],
                   [2 / scale, 0, -1 / scale, 0, 0, -1])
        self.check(Cone(3, 3, 4), [[0, 0, 0], [2, 0, 0]], [3, 1])
        shell = Cone(3, 5, 2, 1, 2)
        inner_scale = np.hypot(1, 0.25)
        self.check(shell, [[0, 0, 0], [1.5, 0, 0], [2, 0, 0]],
                   [-1.5 / inner_scale, 0, 0.5 / inner_scale])
        self.check(Cone(0, 2, 1), [[0, 0, -1], [0, 0, 0], [0, 0, -2]],
                   [0, 1 / np.sqrt(2), -1])
        self.check(Cone(2, 0, 1), [[0, 0, 1]], [0])
        self.check(Cone(0, 3, 1, 0, 1), [[0, 0, -1], [0, 0, 0]],
                   [0, -0.5 / np.hypot(1, 0.5)])

    def test_sector_minor_major_wrap_and_axis(self):
        ball = tsnav.Sphere(10)
        self.check(Sector(ball, 0, np.pi / 2),
                   [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [0, 0, 0], [1, 0, 0]],
                   [1, -1, -1, 0, 0])
        self.check(Sector(ball, 0, 3 * np.pi / 2),
                   [[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0], [0, 0, 0]],
                   [1, 1, 1, -1, 0])
        self.check(Sector(ball, 0, np.pi), [[0, 1, 0], [0, -1, 0]], [1, -1])
        wrapped = Sector(ball, -np.pi / 4, np.pi / 2)
        self.assertAlmostEqual(wrapped.startphi, 7 * np.pi / 4)
        self.check(wrapped, [[1, 0, 0], [-1, 0, 0], [0, 0, 0]],
                   [1 / np.sqrt(2), -1 / np.sqrt(2), 0])
        full = Sector(ball, 123, 2 * np.pi)
        points = [[0, 0, 0], [1, -1, 0], [-2, 3, 1]]
        np.testing.assert_array_equal(full.clearance(points), ball.clearance(points))

    def test_quarter_torus(self):
        self.check(Sector(tsnav.Torus(4, 1), 0, np.pi / 2),
                   [[4, 0, 0], [0, 4, 0], [-4, 0, 0], [0, -4, 0],
                    [np.sqrt(8), np.sqrt(8), 0], [0, 0, 0]],
                   [0, 0, -4, -4, 1, -3])

    def test_concave_polygon_distance_and_caps(self):
        points = [[1, 1, 0], [3, 0.5, 0], [3, 2, 0], [2.5, 1.5, 0],
                  [4.5, 1.5, 0], [2, 1, 0], [1, 2, 1.8], [1, 2, 3]]
        expected = [1, 0.5, -1, -0.5, -np.sqrt(0.5), 0, 0.2, -1]
        for vertices in (STAIR, STAIR[::-1]):
            self.check(Extrusion(vertices, 2), points, expected)
        square = Extrusion([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2]], 3)
        self.check(square, [[1, 1, 0], [1, 0, 0], [-1, -1, 0]],
                   [1, 0, -np.sqrt(2)])

    def test_frozen_and_byte_backed_arrays(self):
        vertices = np.array(STAIR, dtype=float)
        height = np.array(2.)
        polygon = Extrusion(vertices, height)
        cone = Cone(np.array(1.), 3, height)
        sector = Sector(cone, np.array(0.), np.pi)
        vertices[:] = 99
        height[...] = 99
        self.check(polygon, [[1, 1, 0]], [1])
        self.assertEqual(cone.half_length, 2)
        for array in (polygon.vertices, polygon._edges, polygon._lengths,
                      polygon._units, sector._normals):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
            with self.assertRaises(ValueError):
                array.flat[0] = 10
        for solid, name in ((polygon, 'vertices'), (cone, 'rmax1'), (sector, 'startphi')):
            with self.assertRaises(FrozenInstanceError):
                setattr(solid, name, None)

    def test_invalid_geometry(self):
        bad_polygons = [[], [[0, 0], [1, 0]], [[0, 0, 0]] * 3,
                        [[0, 0], [1, 0], [1, 0], [0, 1]],
                        [[0, 0], [1, 0], [2, 0]],
                        [[0, 0], [2, 2], [0, 2], [2, 0]],
                        [[0, 0], [3, 3], [0, 3], [2, 0]],
                        [[0, 0], [2, 0], [1, 0], [1, 1], [0, 1]],
                        [[0, 0], [2, 0], [2, 2], [1, 0], [0, 2]],
                        [[0, 0], [1, 0], [0, 1], [0, 0]],
                        [[0, 0], [np.inf, 0], [0, 1]]]
        for vertices in bad_polygons:
            with self.subTest(vertices=vertices), self.assertRaises(ValueError):
                Extrusion(vertices, 1)
        constructors = [lambda: Cone(-1, 2, 1), lambda: Cone(0, 0, 1),
                        lambda: Cone(1, 2, 0), lambda: Cone(1, 2, 1, 1, 0),
                        lambda: Cone(1, 2, 1, 0, 3), lambda: Cone(1, 2, 1, -1),
                        lambda: Cone(0, 1, 1e-320),
                        lambda: Sector(None, 0, 1), lambda: Sector([], 0, 1),
                        lambda: Sector(tsnav.Sphere(1), 0, 0),
                        lambda: Sector(tsnav.Sphere(1), 0, -1),
                        lambda: Sector(tsnav.Sphere(1), 0, 2 * np.pi + 1),
                        lambda: Extrusion(STAIR, 0)]
        for bad in (True, 1j, np.inf, np.nan, [1], '1'):
            constructors.extend([lambda b=bad: Cone(b, 2, 1),
                                 lambda b=bad: Cone(1, 2, 1, b),
                                 lambda b=bad: Sector(tsnav.Sphere(1), b, 1),
                                 lambda b=bad: Sector(tsnav.Sphere(1), 0, b),
                                 lambda b=bad: Extrusion(STAIR, b)])
        for construct in constructors:
            with self.assertRaises(ValueError):
                construct()

    def test_query_contract_and_children(self):
        solids = [Cone(1, 3, 2), Extrusion(STAIR, 2), Sector(tsnav.Sphere(2), 0, 1)]
        for solid in solids:
            self.check(solid, np.empty((0, 3)), [])
            for bad in ([], [1, 2, 3], [[0, 0]], [[np.nan, 0, 0]],
                        [[np.inf, 0, 0]], [[True, False, True]], [[1j, 0, 0]]):
                with self.assertRaises(ValueError):
                    solid.clearance(bad)
        self.check(Sector(lambda p: np.ones(len(p)), 0, 2 * np.pi), [[0, 0, 0]], [1])
        for bad in (1, [[1]], [np.nan], [np.inf], [True], [1j]):
            with self.assertRaises(ValueError):
                Sector(lambda p, b=bad: b, 0, 1).clearance([[0, 0, 0]])
        with self.assertRaises(ValueError):
            Sector(lambda p: [0], 0, 2 * np.pi).clearance(np.empty((0, 3)))
        huge = np.finfo(float).max
        with self.assertRaises(ValueError):
            Extrusion([[-huge, 0], [huge, 0], [0, 1]], 1)
        with self.assertRaises(ValueError):
            Extrusion(STAIR, 2).clearance([[huge, huge, 0]])

    def test_deterministic_lipschitz(self):
        rng = np.random.default_rng(512)
        a = rng.uniform(-5, 5, (600, 3))
        b = a + rng.normal(size=a.shape) * 0.3
        solids = [Cone(1, 4, 3), Cone(3, 4, 3, 0, 2), Cone(0, 4, 2),
                  Extrusion(STAIR, 2), Extrusion(STAIR[::-1], 2),
                  Sector(tsnav.Torus(4, 1), 0, np.pi / 2),
                  Sector(tsnav.Sphere(5), 5.5, 1.5),
                  Sector(tsnav.Sphere(5), 0.2, 4.5)]
        for solid in solids:
            delta = np.abs(solid.clearance(a) - solid.clearance(b))
            self.assertTrue(np.all(delta <= np.linalg.norm(a - b, axis=1) + 1e-12))


if __name__ == '__main__':
    unittest.main()


def test_underflowed_nonzero_slope_is_not_a_cylinder():
    import pytest

    with pytest.raises(ValueError, match='slope underflows'):
        Cone(0., 1e-300, 1e100)
