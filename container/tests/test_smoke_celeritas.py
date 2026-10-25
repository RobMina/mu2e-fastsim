"""Check synthetic smoke inputs and controls without a Celeritas installation."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


spec = importlib.util.spec_from_file_location(
    "smoke_celeritas", Path(__file__).resolve().parents[1] / "smoke_celeritas.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTest(unittest.TestCase):
    def test_input(self):
        for pdg in (11, -11, 22):
            card = smoke.make_input(pdg, 64, 1)
            self.assertEqual(card["sd_type"], "simple_calo")
            self.assertEqual(card["primary_options"]["pdg"], [pdg])
            self.assertEqual(card["primary_options"]["num_events"], 64)
            self.assertEqual(card["primary_options"]["energy"]["params"], [10.0])
            self.assertEqual(card["field"], [0, 0, 1])

    def test_bounds(self):
        for pdg, events, field in ((13, 64, 1), (11, 0, 1), (11, 1025, 1), (11, 1, 2)):
            with self.assertRaises(ValueError):
                smoke.make_input(pdg, events, field)

    def test_control_environment(self):
        with patch.dict(os.environ, {"CELER_DISABLE": "1", "CELER_KILL_OFFLOAD": "1",
                                     "CELER_NONFATAL_FLUSH": "1", "CELER_DISABLE_DEVICE": "1"}):
            gpu = smoke.clean_environment("gpu")
            self.assertNotIn("CELER_DISABLE", gpu)
            self.assertNotIn("CELER_KILL_OFFLOAD", gpu)
            self.assertNotIn("CELER_NONFATAL_FLUSH", gpu)
            self.assertNotIn("CELER_DISABLE_DEVICE", gpu)
            self.assertEqual(gpu["G4FORCENUMBEROFTHREADS"], "1")
            self.assertEqual(smoke.clean_environment("g4")["CELER_DISABLE"], "1")

    def fixture(self, mode="gpu"):
        return {
            "input": smoke.make_input(11, 2, 1),
            "result": {"time": {"events": [[0.1, 0.2]], "total": 0.3,
                                "actions": [{"along-step-uniform-msc": 0.1}]},
                       "smoke_calo": {"_units": {"energy_deposition": "MeV"},
                                      "volume_labels": ["absorber"], "energy_deposition": [20.0]}},
            "system": {"device": {"platform": "CUDA", "name": "NVIDIA A100"},
                       "kernels": [{"name": "extend-from-primaries"}, {"name": "step-gather-post"}],
                       "environ": {"CELER_DISABLE": "0" if mode == "gpu" else "1"}},
        }

    def test_inspect_valid(self):
        for mode in ("g4", "gpu"):
            row = smoke.inspect_result(self.fixture(mode), 11, mode, 2)
            self.assertEqual(row["completed_event_records"], 2)
            self.assertEqual(row["relative_energy_error"], 0)

    def test_incomplete_events(self):
        with self.assertRaises(ValueError):
            smoke.inspect_result(self.fixture(), 11, "gpu", 3)

    def test_bad_energy(self):
        for energy in (0, float("nan"), float("inf"), 22.0):
            data = self.fixture()
            data["result"]["smoke_calo"]["energy_deposition"] = [energy]
            with self.assertRaises(ValueError):
                smoke.inspect_result(data, 11, "gpu", 2)

    def test_no_device_or_offload(self):
        data = self.fixture()
        data["system"]["device"]["platform"] = "CPU"
        with self.assertRaises(ValueError):
            smoke.inspect_result(data, 11, "gpu", 2)
        data = self.fixture("g4")
        with self.assertRaises(ValueError):
            smoke.inspect_result(data, 11, "gpu", 2)

    def test_wrong_score(self):
        data = self.fixture()
        data["result"]["smoke_calo"]["_units"]["energy_deposition"] = "GeV"
        with self.assertRaises(ValueError):
            smoke.inspect_result(data, 11, "gpu", 2)

    def test_positron_annihilation_energy(self):
        data = self.fixture()
        data["input"]["primary_options"]["pdg"] = [-11]
        data["result"]["smoke_calo"]["energy_deposition"] = [2 * (10 + 2 * 0.51099891)]
        self.assertEqual(smoke.inspect_result(data, -11, "gpu", 2)["relative_energy_error"], 0)

    def test_sensitive_geometry(self):
        root = ET.fromstring(smoke.GEOMETRY)
        volume = root.find("structure/volume[@name='absorber']")
        self.assertEqual(volume.find("auxiliary").attrib,
                         {"auxtype": "SensDet", "auxvalue": "smoke_calo"})
        self.assertEqual(root.find("solids/box[@name='absorber_box']").get("lunit"), "cm")


if __name__ == "__main__":
    unittest.main()
