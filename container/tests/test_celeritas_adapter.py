"""Unit/contract tests only: these do not compile or execute the C++ adapter."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np


ADAPTER = Path(__file__).resolve().parents[1] / "celeritas_adapter"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ADAPTER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


field = load("field_contract")
prepare = load("prepare")


class FieldContractTest(unittest.TestCase):
    def test_constant(self):
        data = field.synthetic(constant=True)
        self.assertEqual(field.validate(data).shape, (5, 6, 7, 3))
        np.testing.assert_allclose(field.interpolate(data, [[0, 0, 0], [5, -12, 60]]), [[0, 0, 1]] * 2)

    def test_affine_axis_order(self):
        points = np.array([[10, 50, 100], [-800, 500, -1000], [333, 555, 777], [0, 0, 0]])
        matrix = np.array([[1e-5, 2e-5, 3e-5], [4e-5, -2e-5, 6e-5], [7e-5, 8e-5, 1e-5]])
        np.testing.assert_allclose(field.interpolate(field.synthetic(), points),
                                   points @ matrix + [0, 0, 1], atol=1e-15)

    def test_world_margin(self):
        data = field.synthetic()
        field.require_world_inside(data, [-1000] * 3, [1000] * 3)
        for lower, upper in (([-1001] * 3, [1000] * 3), ([-1000] * 3, [2000] * 3),
                             ([0] * 3, [0] * 3), ([float("nan")] * 3, [100] * 3)):
            with self.assertRaises(ValueError):
                field.require_world_inside(data, lower, upper)

    def test_wrong_contract(self):
        for key, value in (("field_unit", "gauss"), ("length_unit", "cm"),
                           ("coordinates", "mu2e"), ("outside_policy", "zero"),
                           ("order", "zyx-component")):
            data = field.synthetic()
            data[key] = value
            with self.assertRaises(ValueError):
                field.validate(data)

    def test_bad_axes(self):
        for key, value in (("num", 2), ("num", 3.0), ("num", True), ("max_mm", -2000),
                           ("min_mm", float("nan")), ("min_mm", "-2000")):
            data = field.synthetic()
            data["axes"][0][key] = value
            with self.assertRaises(ValueError):
                field.validate(data)

    def test_bad_fields(self):
        for value in (float("nan"), float("inf"), 1e40, "1.0"):
            data = field.synthetic()
            data["field"][0] = value
            with self.assertRaises(ValueError):
                field.validate(data)
        data = field.synthetic()
        data["field"].pop()
        with self.assertRaises(ValueError):
            field.validate(data)

    def test_refuses_outside_points(self):
        for p in ([[2000, 0, 0]], [[-2001, 0, 0]], [[float("nan"), 0, 0]]):
            with self.assertRaises(ValueError):
                field.interpolate(field.synthetic(), p)

    def test_float_transform_overflow(self):
        for lo, hi in ((1e50, 2e50), (0, 1e-50)):
            data = field.synthetic()
            data["axes"][0].update(min_mm=lo, max_mm=hi)
            with self.assertRaises(ValueError):
                field.validate(data)

    def test_float_resolution(self):
        data = field.synthetic()
        data["axes"][0].update(min_mm=1e12, max_mm=1e12 + 1)
        with self.assertRaises(ValueError):
            field.validate(data)


class PrepareTest(unittest.TestCase):
    def test_nonunique_anchor(self):
        for text in ("none", "xx"):
            with self.assertRaises(ValueError):
                prepare.replace_once(text, "x", "y")

    def test_rejects_wrong_hash_before_writing(self):
        pin = json.loads((ADAPTER / "upstream-files.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / next(iter(pin["sha256"]))
            first.parent.mkdir(parents=True)
            first.write_text("wrong source")
            with self.assertRaises(ValueError):
                prepare.prepare(root, root / "out")
            self.assertFalse((root / "out").exists())

    @unittest.skipUnless(os.environ.get("M2FS_CELERITAS_SOURCE"), "set pinned upstream source for preparation test")
    def test_pinned_source_preparation(self):
        source = Path(os.environ["M2FS_CELERITAS_SOURCE"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "app"
            prepare.prepare(source, out)
            detector = (out / "app/DetectorConstruction.cc").read_text()
            self.assertIn("CartMapFieldAlongStepFactory", detector)
            self.assertIn("require_world_inside", detector)
            events = (out / "app/EventAction.cc").read_text()
            self.assertLess(events.index("transport_->Flush()"), events.index("m2fs::write_event"))
            card = (out / "app/RunInputIO.json.cc").read_text()
            self.assertEqual(card.count('v.field_type == "cartmap"'), 2)
            self.assertTrue((out / "COPYRIGHT").is_file())
            self.assertTrue((out / "adapter-manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                prepare.prepare(source, out)


if __name__ == "__main__":
    unittest.main()
