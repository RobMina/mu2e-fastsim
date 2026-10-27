"""Header transform and world coverage contracts using synthetic files."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1] / "celeritas_adapter"
SPEC = importlib.util.spec_from_file_location("coverage", HERE / "map_coverage.py")
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


class CoverageTest(unittest.TestCase):
    def header(self, prefix="", suffix=""):
        return prefix + "grid X0=-100 Y0=0 Z0=-100 nX=21 nY=11 nZ=21 dX=10 dY=10 dZ=10\n" + suffix

    def test_declared_offset_symmetry_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.header"
            path.write_text(self.header("# Origin shift for Mu2e: 0 0 0\n", "extendY flip=By\ndata\n"))
            result = coverage.parse_header(path)
            self.assertEqual(result["full_lower_mu2e_mm"], [-100, -100, -100])
            self.assertEqual(result["full_dimensions"], [21, 21, 21])
            self.assertEqual(result["full_scalar_count"], 3 * 21**3)
            path.write_text(self.header())
            result = coverage.parse_header(path)
            self.assertEqual(result["stored_lower_mu2e_mm"], [3804, 0, -8029])
            self.assertFalse(result["origin_declared"])
            for text in (self.header(suffix="extendX flip=Bx\n"), self.header().replace("nX=21", "nX=21.5")):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    coverage.parse_header(path)

    def test_world_transform_and_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header, gdml = root / "test.header", root / "test.gdml"
            header.write_text(self.header("# Origin shift for Mu2e: 0 0 0\n", "extendY flip=By\n"))
            gdml.write_text('<gdml><solids><box name="box" x="18" y="18" z="18" lunit="cm"/></solids>'
                            '<structure><volume name="world"><solidref ref="box"/></volume></structure>'
                            '<setup><world ref="world"/></setup></gdml>')
            result = coverage.audit([header], gdml, [0, 0, 0])
            self.assertEqual(result["verdict"], "covered")
            self.assertEqual(result["maps"][0]["world_margin_mm"], [0, 0, 0])
            self.assertEqual(coverage.audit([header], gdml, [1, 0, 0])["verdict"], "uncovered")


if __name__ == "__main__":
    unittest.main()
