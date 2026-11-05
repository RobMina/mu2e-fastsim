"""Synthetic clearance checks, not a native geometry or transport comparison."""

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from beamfastsim.tsnav import (
    Box, Difference, HalfSpace, Intersection, Placed, Sphere, Torus, Tube, Union,
)


class ClearanceTests(unittest.TestCase):
    def assert_clearance(self, solid, points, expected):
        actual = solid.clearance(points)
        self.assertEqual(actual.shape, (len(points),))
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_analytic_primitives(self):
        self.assert_clearance(Sphere(2), [[0, 0, 0], [2, 0, 0], [0, 3, 4]],
                              [2, 0, -3])
        self.assert_clearance(Box([1, 2, 3]),
                              [[0, 0, 0], [1, 0, 0], [2, 3, 0]], [1, 0, -1])
        self.assert_clearance(Tube(3, 4),
                              [[0, 0, 0], [3, 0, 0], [0, 0, 4], [0, 0, 5]],
                              [3, 0, 0, -1])
        self.assert_clearance(Tube(3, 4, 1),
                              [[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
                              [-1, 0, 1, -1])
        self.assert_clearance(Torus(4, 1),
                              [[4, 0, 0], [-4, 0, 0], [0, -4, 0],
                               [4, 0, 1], [0, 0, 0], [6, 0, 0]],
                              [1, 1, 1, 0, -3, -1])
        self.assert_clearance(HalfSpace([0, 0, 2], 6),
                              [[0, 0, 0], [5, -5, 3], [0, 0, 4]], [3, 0, -1])

    def test_csg_hole_and_slit(self):
        block = Box([3, 3, 3])
        hole = Difference(block, Tube(1, 4))
        self.assert_clearance(hole, [[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
                              [-1, 0, 1, -1])
        slit = Difference(block, Box([0.25, 4, 4]))
        self.assert_clearance(slit, [[0, 2, 0], [0.25, 2, 0], [0.5, 2, 0]],
                              [-0.25, 0, 0.25])
        pair = Union(HalfSpace([1, 0, 0], -1), HalfSpace([-1, 0, 0], -1))
        self.assert_clearance(pair, [[-2, 0, 0], [0, 0, 0], [2, 0, 0]], [1, -1, 1])
        clipped = Intersection(Sphere(2), HalfSpace([0, 0, 1], 0))
        self.assert_clearance(clipped, [[0, 0, -1], [0, 0, 1], [0, 0, 0]], [1, -1, 0])

    def test_rotation_and_translation(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        origin = np.array([10., 20, 30])
        local = np.array([[0., 0, 0], [0.5, 1.5, 0], [2, 0, 0]])
        box = Box([1, 2, 3])
        placed = Placed(box, origin, rotation)
        global_points = local @ rotation.T + origin
        np.testing.assert_allclose(placed.clearance(global_points), box.clearance(local))
        tilted = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
        self.assert_clearance(Placed(Tube(1, 3), [0, 0, 0], tilted),
                              [[2, 0, 0], [0, 0, 2]], [1, -1])

    def test_parameters_copied_and_frozen(self):
        lengths = np.array([1., 2, 3])
        normal = np.array([0., 0, 2])
        origin = np.array([1., 0, 0])
        rotation = np.eye(3)
        radius = np.array(2.)
        box = Box(lengths)
        plane = HalfSpace(normal, 2)
        placed = Placed(box, origin, rotation)
        sphere = Sphere(radius)
        lengths[:] = 99
        normal[:] = 0
        origin[:] = 100
        rotation[:] = 0
        radius[...] = 99
        self.assert_clearance(box, [[0, 0, 0]], [1])
        self.assert_clearance(plane, [[0, 0, 0]], [1])
        self.assert_clearance(placed, [[1, 0, 0]], [1])
        self.assert_clearance(sphere, [[0, 0, 0]], [2])
        for array in (box.half_lengths, plane.normal, placed.origin, placed.rotation):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
            with self.assertRaises(ValueError):
                array.flat[0] = 9
        for solid, name in ((sphere, "radius"), (box, "half_lengths"),
                            (Intersection(box), "solids"), (placed, "origin")):
            with self.assertRaises(FrozenInstanceError):
                setattr(solid, name, None)

    def test_invalid_parameters(self):
        constructors = [
            lambda: Sphere(0), lambda: Sphere(-1), lambda: Sphere([1]),
            lambda: Sphere(True), lambda: Sphere("2"), lambda: Sphere(1 + 0j),
            lambda: Sphere(np.nan), lambda: Sphere(np.inf), lambda: Sphere(10**400),
            lambda: Box([1, 2]), lambda: Box([[1, 2, 3]]), lambda: Box([1, 0, 2]),
            lambda: Box([1, np.inf, 2]), lambda: Box([1, 2j, 3]),
            lambda: Tube(1, 0), lambda: Tube(0, 1), lambda: Tube(1, 1, -1),
            lambda: Tube(1, 1, 1), lambda: Tube(1, 1, [0]),
            lambda: Torus(1, 1), lambda: Torus(1, 2), lambda: Torus(2, 0),
            lambda: Torus([2], 1), lambda: HalfSpace([0, 0, 0], 0),
            lambda: HalfSpace([1, 0], 0), lambda: HalfSpace([1, 0, 0], [0]),
            lambda: HalfSpace([np.inf, 0, 0], 0), lambda: Intersection(),
            lambda: Union(), lambda: Intersection(None), lambda: Union(4),
            lambda: Difference(Sphere(1), None),
            lambda: Placed(None, [0, 0, 0], np.eye(3)),
            lambda: Placed(Sphere(1), [0, 0], np.eye(3)),
            lambda: Placed(Sphere(1), [0, 0, 0], np.ones((2, 2))),
            lambda: Placed(Sphere(1), [0, 0, 0], np.diag([-1, 1, 1])),
            lambda: Placed(Sphere(1), [0, 0, 0], np.diag([1, 1, 2])),
            lambda: Placed(Sphere(1), [0, 0, 0], np.eye(3, dtype=complex)),
            lambda: Placed(Sphere(1), [0, 0, 0], np.full((3, 3), np.inf)),
        ]
        for construct in constructors:
            with self.subTest(construct=construct):
                with self.assertRaises(ValueError):
                    construct()

    def test_invalid_query_types_and_shapes(self):
        invalid = [None, 1, [], [1, 2, 3], [[1, 2]], np.zeros((1, 1, 3)),
                   [[np.nan, 0, 0]], [[0, np.inf, 0]], [[1j, 0, 0]],
                   [["1", "2", "3"]], [[True, False, True]],
                   np.array([[1, 2, 3]], dtype=object)]
        solids = [Sphere(1), Box([1, 1, 1]), Tube(2, 1), Torus(2, 1),
                  HalfSpace([1, 0, 0], 0), Intersection(Sphere(1)),
                  Union(Sphere(1)), Difference(Sphere(2), Sphere(1)),
                  Placed(Sphere(1), [0, 0, 0], np.eye(3))]
        for solid in solids:
            for points in invalid:
                with self.subTest(solid=type(solid).__name__, points=points):
                    with self.assertRaises(ValueError):
                        solid.clearance(points)
            empty = solid.clearance(np.empty((0, 3)))
            self.assertEqual(empty.shape, (0,))
            self.assertEqual(empty.dtype, np.dtype(float))

    def test_generic_children_checked_every_query(self):
        class Child:
            def clearance(self, points):
                return np.ones(len(points))

        child = Child()
        solids = [Intersection(child), Union(child), Difference(child, Sphere(2)),
                  Difference(Sphere(2), child), Placed(child, [0, 0, 0], np.eye(3))]
        for solid in solids:
            self.assertEqual(solid.clearance([[0, 0, 0]]).shape, (1,))
        for bad in (1., np.ones((1, 1)), [np.nan], [np.inf], [1j], ["1"], [True]):
            child.clearance = lambda p, bad=bad: bad
            for solid in solids:
                with self.assertRaises(ValueError):
                    solid.clearance([[0, 0, 0]])
        self.assert_clearance(Union(lambda p: np.ones(len(p))), [[0, 0, 0]], [1])
        with self.assertRaises(ValueError):
            Union(lambda p: [0]).clearance(np.empty((0, 3)))

    def test_overflow_fails_closed(self):
        huge = np.finfo(float).max
        cases = [
            lambda: Sphere(1).clearance([[huge, huge, huge]]),
            lambda: HalfSpace([1, 0, 0], huge).clearance([[-huge, 0, 0]]),
            lambda: HalfSpace([huge, huge, huge], 0),
            lambda: HalfSpace([1e-320, 0, 0], 1),
            lambda: Placed(Sphere(1), [-huge, 0, 0], np.eye(3)).clearance([[huge, 0, 0]]),
            lambda: Placed(Sphere(1), [0, 0, 0], np.full((3, 3), huge)),
        ]
        for evaluate in cases:
            with self.assertRaises(ValueError):
                evaluate()

    def test_lipschitz_and_conservative_not_exact_distance(self):
        rng = np.random.default_rng(123)
        a = rng.uniform(-5, 5, (100, 3))
        b = rng.uniform(-5, 5, (100, 3))
        primitives = [Sphere(2), Box([1, 2, 3]), Tube(3, 4, 1), Tube(3, 4),
                      Torus(3, 1), HalfSpace([1, 2, 3], 2)]
        solids = primitives + [Union(*primitives), Intersection(*primitives),
                               Difference(primitives[0], primitives[1])]
        for solid in solids:
            delta = np.abs(solid.clearance(a) - solid.clearance(b))
            self.assertTrue(np.all(delta <= np.linalg.norm(a - b, axis=1) + 1e-12))
        corner = Box([1, 1, 1]).clearance([[2, 2, 0]])[0]
        self.assertEqual(corner, -1)
        self.assertLess(abs(corner), np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
