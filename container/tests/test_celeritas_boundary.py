"""Synthetic oracle/guard tests only; mock hits are not transport evidence."""

import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parents[1] / "celeritas_adapter"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("boundary", HERE / "boundary.py")
boundary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boundary)


class BoundaryTest(unittest.TestCase):
    def test_charge_and_field_oracles(self):
        negative = boundary.oracle(11, 1)
        positive = boundary.oracle(-11, 1)
        self.assertAlmostEqual(negative["position_mm"][0], positive["position_mm"][0])
        self.assertAlmostEqual(negative["position_mm"][1], -positive["position_mm"][1])
        self.assertGreater(negative["position_mm"][1], 0)
        self.assertEqual(boundary.oracle(11, 0)["position_mm"], [15, 0, 20])
        self.assertAlmostEqual(negative["time_s"], 8.349e-11, delta=1e-14)

    def test_geometry_and_case_isolation(self):
        self.assertEqual(len(boundary.cases()), 40)
        self.assertEqual(len({r["name"] for r in boundary.cases()}), 40)
        for row in boundary.cases():
            root = ET.fromstring(boundary.geometry(row["split_mm"]))
            boxes = root.findall("./solids/box")
            self.assertEqual(len(boxes), 3)
            self.assertAlmostEqual(sum(float(b.attrib["y"]) for b in boxes[1:]), 198)
            self.assertTrue(all(float(b.attrib["z"]) == 0.01 for b in boxes[1:]))
        for axis in ("chord", "delta", "epsilon"):
            self.assertEqual(set(boundary.SETTINGS[f"{axis}_loose"]), set(boundary.SETTINGS[f"{axis}_fine"]))
            self.assertEqual(len(boundary.SETTINGS[f"{axis}_fine"]), 1)

    def fixture(self):
        case = boundary.cases()[0]
        expected = boundary.oracle(case["pdg"], case["field"])
        copy_num = 1 if expected["position_mm"][1] < case["split_mm"] else 2
        rows = [dict(format="m2fs-event-total-edep-v1", event_id=i, primary_count=1, hit_count=1,
                     total_edep_mev=0.01, collections=[dict(sd="surface", total_edep_mev=0.01,
                         hits=[dict(copy_num=copy_num, pre_time_s=expected["time_s"], total_edep_mev=0.01)])])
                for i in range(boundary.EVENTS)]
        return case, rows

    def test_wrong_membership_time_counts_and_energy_reject(self):
        case, rows = self.fixture()
        self.assertTrue(boundary.check_events(rows, case)["passed"])
        bad = copy.deepcopy(rows)
        bad[0]["collections"][0]["hits"][0]["copy_num"] = 3
        self.assertFalse(boundary.check_events(bad, case)["passed"])
        bad = copy.deepcopy(rows)
        bad[0]["collections"][0]["hits"][0]["pre_time_s"] += 1e-12
        self.assertFalse(boundary.check_events(bad, case)["passed"])
        for key, val in (("primary_count", 2), ("event_id", 1), ("total_edep_mev", 0.02), ("hit_count", 0)):
            bad = copy.deepcopy(rows)
            bad[0][key] = val
            with self.assertRaises(ValueError):
                boundary.check_events(bad, case)


if __name__ == "__main__":
    unittest.main()
