"""Offline oracle tests only: never manufacture a successful Celeritas/CUDA result or execute transport."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


SPEC = importlib.util.spec_from_file_location("field_probe_oracle", Path(__file__).resolve().parents[1] / "celeritas_adapter/field_probe/oracle.py")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


class AnalyticOracleTest(unittest.TestCase):
    def test_cgs_fixed_step_count_includes_roundoff_remainder(self):
        self.assertEqual(oracle.fixed_step_count(600, 0.25), 2401)
        self.assertEqual(oracle.fixed_step_count(180, 0.25), 720)
        self.assertEqual(oracle.fixed_step_count(600, 1), 600)
        self.assertEqual(oracle.fixed_step_count(180, 4), 45)
        self.assertEqual(oracle.fixed_step_count(181, 4), 46)
        for length, cap in ((0, 1), (1, 0), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                oracle.fixed_step_count(length, cap)

    def test_absolute_radius_from_si_units(self):
        radius_mm = (25e6 * 1.602176634e-19 / 299792458) / 1.602176634e-19 * 1000
        length_mm = radius_mm * np.pi / 2
        x, p = oracle.constant_helix([[0, 0, 0]], [[25, 0, 0]], 1, length_mm, [0, 0, 1])
        np.testing.assert_allclose(x, [[radius_mm, -radius_mm, 0]], atol=1e-11)
        np.testing.assert_allclose(p, [[0, -25, 0]], atol=1e-11)

    def test_known_helix_sign_and_neutral(self):
        radius = 25 / oracle.KAPPA
        length = np.pi * radius / 2
        x, p = oracle.constant_helix(np.zeros((3, 3)), [[25, 0, 0]] * 3, [1, -1, 0], length, [0, 0, 1])
        np.testing.assert_allclose(x, [[radius, -radius, 0], [radius, radius, 0], [length, 0, 0]], atol=2e-12)
        np.testing.assert_allclose(p, [[0, -25, 0], [0, 25, 0], [25, 0, 0]], atol=2e-13)

    def test_zero_field_mixed_directions(self):
        positions = np.array([[1, 2, 3], [-7, 4, 1], [11, -3, 9]])
        momenta = np.array([[3, 4, 0], [0, -15, 20], [-35, 70, 70]])
        lengths = np.array([5, 25, 105])
        x, p = oracle.constant_helix(positions, momenta, [-1, 0, 1], lengths, [0, 0, 0])
        np.testing.assert_allclose(x, positions + momenta, atol=2e-14)
        np.testing.assert_allclose(p, momenta, atol=2e-14)

    def test_parallel_and_complete_turn(self):
        radius = 5 / oracle.KAPPA
        x, p = oracle.constant_helix([[0, 0, 0], [1, 2, 3]], [[5, 0, 0], [0, 0, -25]], [1, -1], [2 * np.pi * radius, 100], [0, 0, 1])
        np.testing.assert_allclose(x, [[0, 0, 0], [1, 2, -97]], atol=1e-12)
        np.testing.assert_allclose(p, [[5, 0, 0], [0, 0, -25]], atol=1e-12)

    def test_tiny_rotation_is_stable(self):
        x, p = oracle.constant_helix([[0, 0, 0]], [[5, 0, 0]], 1, 1e-8, [0, 0, 1])
        self.assertGreater(x[0, 0], 0)
        self.assertLess(x[0, 1], 0)
        self.assertLess(p[0, 1], 0)
        self.assertAlmostEqual(x[0, 0], 1e-8, places=20)

    def test_affine_reference_two_precisions_and_batch(self):
        b0, gradient = oracle.model("affine")
        self.assertAlmostEqual(float(np.trace(gradient)), 0.0, places=18)
        positions = [[-30, 70, 20], [100, -120, 50], [0, 0, 0]]
        momenta = [[3, 4, 0], [-15, 0, 20], [35, -70, 70]]
        charge, length = [-1, 1, 0], [180, 600, 600]
        tight_x, tight_p = oracle.affine_ode(positions, momenta, charge, length, b0, gradient)
        loose_x, loose_p = oracle.affine_ode(positions, momenta, charge, length, b0, gradient, tight=False)
        np.testing.assert_allclose(tight_x, loose_x, atol=oracle.REF_POSITION_MM, rtol=0)
        np.testing.assert_allclose(tight_p, loose_p, atol=1e-8, rtol=0)
        single_x, single_p = oracle.affine_ode(positions[:1], momenta[:1], charge[:1], length[:1], b0, gradient)
        np.testing.assert_allclose(tight_x[:1], single_x, atol=1e-7, rtol=0)
        np.testing.assert_allclose(tight_p[:1], single_p, atol=1e-9, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(tight_p, axis=1), np.linalg.norm(momenta, axis=1), atol=1e-9)
        np.testing.assert_allclose(tight_x[2], np.array(momenta[2]) * (600 / 105), atol=1e-10)

    def test_dop853_matches_independent_tilted_helix(self):
        positions = [[-40, 60, 100], [100, 20, -60], [0, 0, 0]]
        momenta = [[3, 4, 0], [-15, 0, 20], [35, -70, 70]]
        b0, gradient = oracle.model("constant")
        analytic = oracle.constant_helix(positions, momenta, [-1, 1, 0], [600, 600, 180], b0)
        numerical = oracle.affine_ode(positions, momenta, [-1, 1, 0], [600, 600, 180], b0, gradient)
        np.testing.assert_allclose(numerical[0], analytic[0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(numerical[1], analytic[1], atol=2e-9, rtol=0)

    def test_invalid_initial_conditions(self):
        for momentum, charge, length in (([0, 0, 0], 1, 10), ([5, 0, 0], 2, 10), ([5, 0, 0], 1, 0), ([5, 0, 0], 1, float("nan"))):
            with self.assertRaises(ValueError):
                oracle.constant_helix([[0, 0, 0]], [momentum], charge, length, [0, 0, 1])


class FixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.request, cls.expected = oracle.build_case("affine", "coarse")

    def test_map_contract_flattening_and_divergence(self):
        data = self.data
        field = oracle._CONTRACT_MODULE.validate(data)
        self.assertEqual(field.shape, (10, 15, 13, 3))
        b0, gradient = oracle.model("affine")
        point = [[data["axes"][0]["min_mm"], data["axes"][1]["max_mm"], data["axes"][2]["min_mm"]]]
        np.testing.assert_allclose(field[0, -1, 0], oracle.field_at(point, b0, gradient)[0], atol=1e-15)
        self.assertEqual(len({a["num"] for a in data["axes"]}), 3)
        self.assertEqual(len({a["max_mm"] - a["min_mm"] for a in data["axes"]}), 3)

    def test_fixture_coverage_and_metadata_separation(self):
        oracle.validate_request(self.request)
        self.assertEqual(len(self.request["sample_points_mm"]), 1016)
        self.assertEqual(len(self.request["tracks"]), 96)
        self.assertEqual(sum(self.expected["sample_valid"]), 968)
        self.assertEqual(sum(t["expected_status"] == 1 for t in self.expected["tracks"]), 6)
        self.assertEqual({t["charge_e"] for t in self.request["tracks"]}, {-1, 0, 1})
        self.assertEqual({t["method"] for t in self.request["tracks"]}, {"fixed", "adaptive"})
        self.assertNotIn("expected", self.request)
        self.assertNotIn("model", self.request)
        for axis in range(3):
            for side in (-1, 1):
                self.assertIn(f"inset_face_{axis}_{side:+d}", self.expected["sample_categories"])
                self.assertIn(f"outside_inset_{axis}_{side:+d}", self.expected["sample_categories"])
        self.assertLess(self.expected["reference_precision"]["position_disagreement_mm"], oracle.REF_POSITION_MM)
        self.assertLess(self.expected["reference_precision"]["momentum_relative_disagreement"], oracle.REF_MOMENTUM_REL)

    def test_affine_quantization_bound_refines_without_fitting(self):
        b0, gradient = oracle.model("affine")
        host, device = oracle.field_error_bounds(self.data, b0, gradient)
        fine_host, fine_device = oracle.field_error_bounds(oracle.make_map("affine", "refined"), b0, gradient)
        lo, hi, n, spacing = oracle.grid_geometry(self.data)
        np.testing.assert_allclose(device - host, spacing @ abs(gradient) / 256, atol=1e-19)
        np.testing.assert_allclose(device - host, 2 * (fine_device - fine_host), atol=1e-19)
        self.assertTrue(np.all(host > 0))
        self.assertTrue(np.all(fine_host <= host))

    def test_all_models_and_resolutions_generate_without_execution(self):
        for family in oracle.FAMILIES:
            for resolution in oracle.RESOLUTIONS:
                data = oracle.make_map(family, resolution)
                b0, gradient = oracle.model(family)
                h, d = oracle.field_error_bounds(data, b0, gradient)
                self.assertTrue(np.isfinite(h).all() and np.isfinite(d).all())
                if family != "affine":
                    np.testing.assert_array_equal(h, d)
                points, categories = oracle.sample_points(data)
                self.assertEqual(points.shape, (1016, 3))
                self.assertEqual(len(categories), len(points))

    def test_reference_reproducible_and_sweeps_share_truth(self):
        _, request, expected = oracle.build_case("affine", "coarse")
        self.assertEqual(request, self.request)
        self.assertEqual(expected, self.expected)
        for offset in range(0, 90, 6):
            endpoints = [t["end_position_mm"] for t in expected["tracks"][offset:offset + 6]]
            self.assertTrue(all(endpoint == endpoints[0] for endpoint in endpoints))

    def test_prepare_refuses_existing_path_without_writing(self):
        with self.assertRaisesRegex(ValueError, "NEW_OUTPUT"):
            oracle.prepare(Path(__file__).parent)

    def test_prepare_all_cases_in_memory_without_writes(self):
        with mock.patch.object(Path, "exists", return_value=False), mock.patch.object(Path, "mkdir"), mock.patch.object(oracle, "write_json") as writer:
            index = oracle.prepare("NOT_A_REAL_OUTPUT_DIRECTORY")
        self.assertEqual(len(index["cases"]), 6)
        self.assertEqual(writer.call_count, 19)
        self.assertIn("Not run", index["execution"])
        for call in writer.call_args_list:
            json.dumps(call.args[1], allow_nan=False)

    def test_bad_request(self):
        for key, value in (("tracks", []), ("sample_points_mm", []), ("map_file", "../escape.json"), ("format", "wrong")):
            request = copy.deepcopy(self.request)
            request[key] = value
            with self.assertRaises(ValueError):
                oracle.validate_request(request)
        request = copy.deepcopy(self.request)
        request["tracks"][1]["id"] = request["tracks"][0]["id"]
        with self.assertRaises(ValueError):
            oracle.validate_request(request)


class FailClosedCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.request, cls.expected = oracle.build_case("constant", "coarse")

    def incomplete_envelope(self):
        # This deliberately incomplete mock cannot pass; it is not claimed as real Celeritas evidence.
        return dict(format=oracle.RESULT_FORMAT, input=copy.deepcopy(self.request), celeritas_version="0.6.3", device=dict(name="NVIDIA A100 TEST MOCK NOT EVIDENCE", capability=80, synchronized=True), samples=dict(host=[{} for _ in self.request["sample_points_mm"]], device=[{} for _ in self.request["sample_points_mm"]]), tracks=dict(host=[{} for _ in self.request["tracks"]], device=[{} for _ in self.request["tracks"]]))

    def incomplete_with_host_samples(self):
        # Populate only host sample mocks to reach negative track checks; all track/device records stay incomplete.
        result = self.incomplete_envelope()
        result["samples"]["host"] = [dict(field_tesla=field if valid else [None, None, None], valid=valid, field_evaluations=int(valid), guard_violations=int(not valid), numerical_failures=0, min_margin_mm=margin, completed_tag=oracle.TAG) for field, valid, margin in zip(self.expected["sample_field_tesla"], self.expected["sample_valid"], self.expected["sample_margin_mm"])]
        return result

    def assert_rejected(self, result, pattern=None):
        context = self.assertRaisesRegex(ValueError, pattern) if pattern else self.assertRaises(ValueError)
        with context:
            oracle.check_result(self.request, result, self.expected)

    def test_malformed_or_empty_result(self):
        for result in ({}, [], None, "", {"format": oracle.RESULT_FORMAT}):
            self.assert_rejected(result)
        result = self.incomplete_envelope()
        result["samples"]["device"] = []
        self.assert_rejected(result, "missing/extra")
        result = self.incomplete_envelope()
        result["tracks"]["host"] = []
        self.assert_rejected(result, "missing/extra")

    def test_fallback_unsynchronized_and_wrong_version(self):
        for key, value in (("name", "CPU fallback"), ("capability", 0), ("synchronized", False), ("synchronized", 1)):
            result = self.incomplete_envelope()
            result["device"][key] = value
            self.assert_rejected(result, "fallback|capability|synchronized")
        result = self.incomplete_envelope()
        result["celeritas_version"] = "0.6.2"
        self.assert_rejected(result, "version")
        result = self.incomplete_envelope()
        result["fallback"] = True
        self.assert_rejected(result, "keys")

    def test_exact_input_and_sidecar_identity(self):
        result = self.incomplete_envelope()
        result["input"]["tracks"][0]["max_step_mm"] *= 2
        self.assert_rejected(result, "exact request")
        expected = copy.deepcopy(self.expected)
        expected["request_sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "identity"):
            oracle.check_result(self.request, self.incomplete_envelope(), expected)

    def test_missing_completion_tag(self):
        result = self.incomplete_envelope()
        first = dict(field_tesla=[0, 0, 0], valid=True, field_evaluations=1, guard_violations=0, numerical_failures=0, min_margin_mm=self.expected["sample_margin_mm"][0])
        result["samples"]["host"][0] = first
        self.assert_rejected(result, "keys")
        first["completed_tag"] = 0
        self.assert_rejected(result, "completion")
        first["completed_tag"] = True
        self.assert_rejected(result, "completion")

    def test_shape_accounting_nonfinite_and_wrong_validity(self):
        for change, pattern in (({"field_tesla": [0, 0]}, "shape"), ({"field_tesla": [0, 0, float("nan")]}, "JSON"), ({"guard_violations": 1}, "accounting"), ({"valid": False}, "validity"), ({"field_evaluations": True}, "integer"), ({"min_margin_mm": float("inf")}, "JSON")):
            result = self.incomplete_envelope()
            row = dict(field_tesla=[0, 0, 0], valid=True, field_evaluations=1, guard_violations=0, numerical_failures=0, min_margin_mm=self.expected["sample_margin_mm"][0], completed_tag=oracle.TAG)
            row.update(change)
            result["samples"]["host"][0] = row
            self.assert_rejected(result, pattern)

    def test_field_mismatch_and_rejected_query_access(self):
        result = self.incomplete_with_host_samples()
        result["samples"]["host"][0]["field_tesla"] = [4, 5, 6]
        self.assert_rejected(result, "field mismatch")
        result = self.incomplete_with_host_samples()
        rejected = next(i for i, valid in enumerate(self.expected["sample_valid"]) if not valid)
        result["samples"]["host"][rejected]["field_evaluations"] = 1
        self.assert_rejected(result, "rejected query accessed field")

    def test_track_completion_status_path_and_momentum_fail_closed(self):
        track = self.request["tracks"][0]
        meta = self.expected["tracks"][0]
        row = dict(end_position_mm=meta["end_position_mm"], end_momentum_mev_c=meta["end_momentum_mev_c"], path_mm=track["length_mm"], steps=45, field_evaluations=315, guard_violations=0, numerical_failures=0, min_margin_mm=100, momentum_relative_drift=0, max_step_mm_used=track["max_step_mm"], status=0, completed_tag=oracle.TAG)
        for change, pattern in (({"completed_tag": 0}, "completion"), ({"status": 1}, "status"), ({"status": 2}, "status"), ({"status": 3}, "status"), ({"path_mm": 0}, "incomplete"), ({"guard_violations": 1}, "field access"), ({"field_evaluations": 0}, "evaluation accounting"), ({"steps": 1}, "path/step accounting"), ({"numerical_failures": 1}, "numerical failure"), ({"end_position_mm": [None, None, None]}, "shape"), ({"end_momentum_mev_c": [1, 1, 1]}, "momentum conservation"), ({"momentum_relative_drift": 0.01}, "momentum conservation"), ({"max_step_mm_used": 1000}, "step cap")):
            result = self.incomplete_with_host_samples()
            result["tracks"]["host"][0] = dict(row, **change)
            self.assert_rejected(result, pattern)
        result = self.incomplete_with_host_samples()
        result["tracks"]["host"][0] = dict(row)
        del result["tracks"]["host"][0]["completed_tag"]
        self.assert_rejected(result, "keys")

    def test_reference_precision_cannot_be_waived(self):
        expected = copy.deepcopy(self.expected)
        expected["reference_precision"]["position_disagreement_mm"] = 1
        with self.assertRaisesRegex(ValueError, "Reference precision"):
            oracle.check_result(self.request, self.incomplete_envelope(), expected)

    def test_incomplete_fixture_index_cannot_pass(self):
        for cases in ([], [{}], [{"name": "affine-coarse"}] * 6):
            with mock.patch.object(oracle, "read_json", return_value=dict(format="m2fs-field-probe-fixtures-v1", cases=cases)):
                with self.assertRaises(ValueError):
                    oracle.check_directory("NOT_A_REAL_OUTPUT_DIRECTORY")

    def test_strict_json_rejects_nonfinite_duplicate_and_empty(self):
        for text in ("", '{"x": NaN}', '{"x": Infinity}', '{"x": 1, "x": 2}'):
            with mock.patch.object(Path, "read_text", return_value=text):
                with self.assertRaises(ValueError):
                    oracle.read_json("NOT_A_REAL_FILE")


class SavedProbeReplayTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("M2FS_FIELD_PROBE_RESULTS"), "set real saved h6 results for replay")
    def test_real_results_pass_and_report_is_strict_json(self):
        report = oracle.check_directory(Path(os.environ["M2FS_FIELD_PROBE_RESULTS"]))
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["cases"]), 6)
        json.dumps(report, allow_nan=False)

    @unittest.skipUnless(os.environ.get("M2FS_FIELD_PROBE_RESULTS"), "set real saved h6 results for replay")
    def test_saved_result_extra_step_and_missing_evaluation_are_rejected(self):
        directory = Path(os.environ["M2FS_FIELD_PROBE_RESULTS"])
        request = oracle.read_json(directory / "zero-coarse.request.json")
        result = oracle.read_json(directory / "zero-coarse.result.json")
        expected = oracle.read_json(directory / "zero-coarse.expected.json")
        index = next(i for i, t in enumerate(request["tracks"])
                     if t["length_mm"] == 600 and t["max_step_mm"] == 0.25)
        for change in ({"steps": 2400, "field_evaluations": 16800},
                       {"steps": 2402, "field_evaluations": 16814},
                       {"field_evaluations": 16806}):
            modified = copy.deepcopy(result)
            modified["tracks"]["host"][index].update(change)
            with self.assertRaisesRegex(ValueError, "fixed-step/evaluation accounting"):
                oracle.check_result(request, modified, expected)


if __name__ == "__main__":
    unittest.main()
