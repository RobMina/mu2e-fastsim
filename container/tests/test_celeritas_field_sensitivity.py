"""Control-isolation fixture and fail-closed response tests; mock metrics are not transport evidence."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parents[1] / "celeritas_adapter/field_probe"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("field_sensitivity", HERE / "sensitivity.py")
sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sensitivity)


class SensitivityTest(unittest.TestCase):
    def test_isolation_counts_references_and_request(self):
        _, request, expected, controls = sensitivity.build_case("constant", "coarse")
        self.assertEqual(len(request["tracks"]), 67)
        self.assertEqual(len(controls), 67)
        self.assertEqual(len(request["sample_points_mm"]), 1016)
        self.assertEqual(sum(expected["sample_valid"]), 968)
        self.assertEqual(sensitivity.oracle.digest(request), expected["request_sha256"])
        for axis, vary in (("epsilon", "epsilon_rel_max"), ("chord", "delta_chord_mm"), ("cap", "max_step_mm")):
            for group in sensitivity.GROUPS:
                selected = [(t, m) for t, m, c in zip(request["tracks"], expected["tracks"], controls)
                            if c["physical_group"] == group and c["axis"] == axis]
                self.assertEqual(len(selected), 4)
                self.assertEqual(len({t[vary] for t, _ in selected}), 4)
                self.assertTrue(all(m["end_position_mm"] == selected[0][1]["end_position_mm"] for _, m in selected))
                for key in ("epsilon_rel_max", "delta_chord_mm", "max_step_mm"):
                    if key != vary:
                        self.assertEqual(len({t[key] for t, _ in selected}), 1)
        json.dumps(expected, allow_nan=False)

    def test_response_gates_reject_plateau_and_inaccurate_endpoint(self):
        def metrics(error, work):
            return dict(position_error_mm=error, direction_error=error / 100,
                        field_evaluations=work, steps=work)
        controlled = {"device": {f"{g}/{axis}": [metrics(0.1, 10), metrics(0.1, 10)]
                                 for g in sensitivity.GROUPS[:2] for axis in ("epsilon", "chord")}}
        self.assertTrue(all(not g["passed"] for g in sensitivity.response_gates(controlled)))
        for rows in controlled["device"].values():
            rows[-1] = metrics(1e-6, 100)
        self.assertTrue(all(g["passed"] for g in sensitivity.response_gates(controlled)))
        for rows in controlled["device"].values():
            rows[-1]["direction_error"] = 0.1
        self.assertTrue(all(not g["passed"] for g in sensitivity.response_gates(controlled)))

    def test_runner_pins_existing_binary_and_contains_no_build(self):
        text = (HERE / "run_sensitivity.sh").read_text()
        self.assertIn("a423c659bda77c78b738fc50c5a63540caecf1807aa19f16beb7a395e10922a6", text)
        self.assertIn("--sanitizer", text)
        self.assertNotIn("cmake --build", text)
        self.assertNotIn("ninja ", text)


if __name__ == "__main__":
    unittest.main()
