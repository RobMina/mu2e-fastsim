"""Offline mathematics/schema tests; minimal mock records are NOT transport evidence."""

import copy
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("navigation_fixtures", ROOT / "celeritas_adapter/navigation_probe/fixtures.py")
nav = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nav)
SPEC = importlib.util.spec_from_file_location("navigation_test_contract", ROOT / "celeritas_adapter/field_contract.py")
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def minimal_negative_mock(kind="record_limit"):
    """One straight mock record exercises rejection paths only, never runtime evidence."""
    track = nav.track_config("MOCK_NOT_TRANSPORT_EVIDENCE", max_step_mm=1.0, max_substeps=1 if kind == "outer_limit" else 1000)
    request = {"format": nav.REQUEST_FORMAT, "map_file": "mock.map.json", "geometry_file": "mock.gdml", "max_records": 1 if kind == "record_limit" else 2048, "tracks": [track]}
    before = {"position_mm": [-10.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0], "volume_id": 0, "instance_id": 0, "boundary": False, "outside": False}
    arrival = dict(before, position_mm=[-9.0, 0.0, 0.0])
    speed = 299.792458 * track["momentum_mev_c"] / math.hypot(track["momentum_mev_c"], 0.5109989461)
    step = {"before": before, "arrival": arrival, "after": copy.deepcopy(arrival), "requested_mm": 1.0, "distance_mm": 1.0, "cumulative_path_mm": 1.0, "momentum_mev_c": 5.0, "time_ns": 1.0 / speed, "field_evaluations": 7, "substep_calls": 1, "boundary": False, "looping": kind == "outer_limit", "geometry_failed": False, "completed_tag": nav.TAG}
    output = {"id": track["id"], "status": 2 if kind == "outer_limit" else 3, "completed_tag": nav.TAG, "count": 1, "crossings": 0, "path_mm": 1.0, "field_evaluations": 7, "substep_calls": 1, "min_margin_mm": 215.0, "final": copy.deepcopy(arrival), "steps": [step]}
    result = {"format": nav.RESULT_FORMAT, "input": copy.deepcopy(request), "backend": "host", "units": {"length": "mm", "time": "ns", "momentum": "MeV/c"}, "metadata": {"logical_volumes": [{"id": 0, "name": "nav_world", "ext": ""}], "volume_instances": [{"id": 0, "name": "nav_world_pv", "ext": ""}]}, "tracks": {"host": [output]}}
    expected = {"format": nav.EXPECTED_FORMAT, "request_sha256": nav.digest(request), "kind": kind, "model": {"b0_tesla": [0.0, 0.0, 0.0], "gradient_tesla_per_mm": np.zeros((3, 3)).tolist()}, "boxes": copy.deepcopy(nav.BOXES), "budgets": dict(nav.BUDGETS), "tracks": [{"id": track["id"], "status": output["status"], "initial_volume": "nav_world"}]}
    return request, result, expected


class IndependentMathematicsTest(unittest.TestCase):
    def test_helix_sign_radius_and_neutral(self):
        radius = 5 / nav.KAPPA
        for charge in (-1, 1):
            x, u = nav.helix([0, 0, 0], [1, 0, 0], 5, charge, [0, np.pi * radius / 2, 2 * np.pi * radius], [0, 0, 1])
            np.testing.assert_allclose(x, [[0, 0, 0], [radius, -charge * radius, 0], [0, 0, 0]], atol=2e-13)
            np.testing.assert_allclose(u, [[1, 0, 0], [0, -charge, 0], [1, 0, 0]], atol=2e-14)
        x, u = nav.helix([1, 2, 3], [0, 0.6, 0.8], 5, 0, [0, 5, 25], [1, 2, 3])
        np.testing.assert_allclose(x, [[1, 2, 3], [1, 5, 7], [1, 17, 23]], atol=1e-14)
        np.testing.assert_allclose(u, [[0, 0.6, 0.8]] * 3, atol=1e-14)

    def test_affine_dop853_and_constant_limit(self):
        constant = {"b0_tesla": [0.1, -0.05, 1.0], "gradient_tesla_per_mm": np.zeros((3, 3)).tolist()}
        track = nav.track_config("oracle", length_mm=80, momentum_mev_c=15)
        epsilon_gradient = np.diag([1e-20, -1e-20, 0])
        numerical = nav.Reference(track, dict(constant, gradient_tesla_per_mm=epsilon_gradient.tolist()))
        analytic = nav.Reference(track, constant)
        paths = np.linspace(0, 80, 1001)
        a, b = numerical(paths), analytic(paths)
        np.testing.assert_allclose(a[0], b[0], atol=2e-10, rtol=0)
        np.testing.assert_allclose(a[1], b[1], atol=2e-12, rtol=0)
        self.assertAlmostEqual(np.trace(nav.GRADIENT), 0, places=18)
        affine = nav.Reference(track, dict(constant, gradient_tesla_per_mm=nav.GRADIENT.tolist()))
        x, u = affine(paths)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1, atol=2e-10)
        self.assertGreater(np.linalg.norm(x[-1] - b[0][-1]), 1e-4)

    def test_repeated_roots_not_lost_to_equal_endpoint_signs(self):
        track = nav.track_config("many_turns", position_mm=[-5, 0, 0], length_mm=230)
        model = {"b0_tesla": [0, 0, 1], "gradient_tesla_per_mm": np.zeros((3, 3)).tolist()}
        reference = nav.Reference(track, model)
        roots = nav.plane_roots(reference, 0, -0.05)
        self.assertEqual(len(roots), 5)
        np.testing.assert_allclose(reference(roots)[0][:, 0], -0.05, atol=2e-10)
        expected = nav.expected_track(track, model)
        self.assertEqual(len(expected["crossings"]), 20)
        self.assertEqual([c["volume_after"] for c in expected["crossings"][:4]], ["nav_slab", "nav_world", "nav_thin", "nav_world"])
        for crossing in expected["crossings"]:
            self.assertAlmostEqual(np.linalg.norm(crossing["normal"]), 1)
            self.assertAlmostEqual(crossing["position_mm"][crossing["face_axis"]], crossing["face_coordinate_mm"], places=8)

    def test_tangent_pair_and_exact_tangent_refusal(self):
        cases = {name: (model, tracks) for name, model, tracks, _, _ in nav.fixture_definitions()}
        model, tracks = cases["tangent"]
        for track in tracks:
            expected = nav.expected_track(track, model)
            self.assertEqual(len(expected["crossings"]), 2 if "hit" in track["id"] else 0)
        tangent = nav.track_config("ambiguous", position_mm=[-5 / nav.KAPPA - 0.05, 0, 0], length_mm=np.pi * 5 / nav.KAPPA)
        with self.assertRaisesRegex(ValueError, "tangency"):
            nav.plane_roots(nav.Reference(tangent, model), 0, -0.05)

    def test_affine_variation_exceeds_constant_substitution_budget(self):
        _, model, tracks, _, _ = next(row for row in nav.fixture_definitions() if row[0] == "affine")
        constant = dict(model, gradient_tesla_per_mm=np.zeros((3, 3)).tolist())
        for track in tracks:
            paths = np.linspace(0, track["length_mm"], 257)
            actual = nav.Reference(track, model)(paths)
            substitute = nav.Reference(track, constant)(paths)
            if track["charge_e"]:
                self.assertGreater(np.max(np.linalg.norm(actual[0] - substitute[0], axis=1)),
                                   2 * nav.AFFINE_BUDGETS["position_mm"])
                self.assertGreater(np.max(np.linalg.norm(actual[1] - substitute[1], axis=1)),
                                   2 * nav.AFFINE_BUDGETS["direction"])
            else:
                np.testing.assert_allclose(actual[0], substitute[0], atol=2e-10)
                charged = nav.Reference(dict(track, charge_e=1), model)(paths)
                self.assertGreater(np.max(np.linalg.norm(actual[0] - charged[0], axis=1)), 1)

    def test_outer_limit_trial_envelope_is_inside_map_guard(self):
        _, model, tracks, kind, _ = next(row for row in nav.fixture_definitions() if row[0] == "outer_limit")
        track = tracks[0]
        self.assertEqual(kind, "outer_limit")
        self.assertEqual(track["max_substeps"], 1)
        stage_rows = ((0.2,), (0.075, 0.225), (44/45, -56/15, 32/9),
                      (19372/6561, -25360/2187, 64448/6561, -212/729),
                      (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
                      (35/384, 0, 500/1113, 125/192, -2187/6784, 11/84))
        bound = max(sum(abs(value) for value in row) for row in stage_rows) * track["max_step_mm"]
        margin = np.min(225 - np.abs(track["position_mm"]))
        self.assertLess(bound, margin)
        radius = track["momentum_mev_c"] / nav.KAPPA
        sagitta = radius * (1 - math.cos(track["max_step_mm"] / (2 * radius)))
        self.assertGreater(sagitta, 100 * track["delta_chord_mm"])

    def test_guard_true_orbit_stays_inside_world(self):
        _, model, tracks, kind, _ = next(row for row in nav.fixture_definitions() if row[0] == "guard")
        self.assertEqual(kind, "guard")
        track = tracks[0]
        points = nav.Reference(track, model)(np.linspace(0, track["length_mm"], 10001))[0]
        self.assertTrue(np.all(nav.volume_at(points) == "nav_world"))
        self.assertLessEqual(points[:, 0].max(), 199.9 + 1e-12)
        self.assertGreater(points[:, 0].min(), 198.5)
        self.assertGreater(track["max_step_mm"], 225 - track["position_mm"][0])


class FixturePreparationTest(unittest.TestCase):
    def test_schema_map_geometry_and_expectations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "NEW"
            index = nav.prepare(root)
            self.assertEqual(len(index["cases"]), 8)
            self.assertEqual(sum(len(json.loads((root / c["request"]).read_text())["tracks"]) for c in index["cases"]), 24)
            for case in index["cases"]:
                request = json.loads((root / case["request"]).read_text())
                expected = json.loads((root / case["expected"]).read_text())
                self.assertIs(nav.validate_request(request), request)
                self.assertEqual(expected["request_sha256"], nav.digest(request))
                self.assertFalse((root / case["result"]).exists())
                self.assertNotIn("backend", request)
                data = json.loads((root / request["map_file"]).read_text())
                values = contract.validate(data)
                self.assertEqual(values.shape, (21, 21, 21, 3))
                contract.require_world_inside(data, nav.WORLD["lower_mm"], nav.WORLD["upper_mm"])
                geometry = ET.fromstring((root / request["geometry_file"]).read_text())
                self.assertEqual({v.attrib["name"] for v in geometry.findall("./structure/volume")}, {"nav_world", "nav_slab", "nav_thin"})
                self.assertEqual(len(geometry.findall("./structure/volume/physvol")), 2)
                if case["kind"] == "positive":
                    self.assertEqual(len(expected["tracks"]), len(request["tracks"]))
                    self.assertTrue(all("final" in track and "crossings" in track for track in expected["tracks"]))
                elif case["kind"] == "guard":
                    self.assertEqual(expected["tracks"], [])
                    self.assertEqual(expected["process_marker"], "M2FS_NAV_MAP_GUARD")
                else:
                    self.assertEqual(expected["tracks"][0]["status"], 2 if case["kind"] == "outer_limit" else 3)
            with self.assertRaisesRegex(ValueError, "NEW"):
                nav.prepare(root)

    def test_selected_preparation_is_explicit_and_rejects_unknown_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = nav.prepare(root / "selected", ["outer_limit", "zero"])
            self.assertTrue(index["selected_fixture_set"])
            self.assertEqual({case["id"] for case in index["cases"]}, {"outer_limit", "zero"})
            with self.assertRaisesRegex(ValueError, "selected cases"):
                nav.prepare(root / "bad", ["typo"])
            self.assertFalse((root / "bad").exists())

    def test_request_limits_and_invalid_types(self):
        request, _, _ = minimal_negative_mock()
        mutations = [("max_records", True), ("max_records", 0), ("max_records", 262145), ("tracks", []), ("map_file", "../escape.json")]
        for key, value in mutations:
            modified = copy.deepcopy(request)
            modified[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                nav.validate_request(modified)
        for key, value in (("direction", [2, 0, 0]), ("charge_e", True), ("momentum_mev_c", 0), ("length_mm", float("inf")), ("delta_intersection_mm", 1e-8), ("max_substeps", 1.5)):
            modified = copy.deepcopy(request)
            modified["tracks"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                nav.validate_request(modified)
        oversized = copy.deepcopy(request)
        oversized["tracks"] = [nav.track_config(str(i)) for i in range(129)]
        with self.assertRaises(ValueError):
            nav.validate_request(oversized)
        defaults = copy.deepcopy(request)
        del defaults["tracks"][0]["max_nsteps"]
        del defaults["tracks"][0]["max_substeps"]
        nav.validate_request(defaults)

    def test_independent_controls(self):
        tracks = next(row[2] for row in nav.fixture_definitions() if row[0] == "constant")
        for charge in (-1, 1):
            baseline = next(t for t in tracks if t["id"] == f"repeat_q{charge}_base")
            for track in (t for t in tracks if t["charge_e"] == charge and not t["id"].endswith("base")):
                changed = {key for key in track if key != "id" and track[key] != baseline[key]}
                self.assertEqual(len(changed), 1)
                self.assertTrue(changed <= {"delta_chord_mm", "delta_intersection_mm", "epsilon_rel_max", "max_step_mm"})


class NegativeCheckerContractTest(unittest.TestCase):
    def test_minimal_mocks_are_only_negative_statuses(self):
        for kind in ("record_limit", "outer_limit"):
            request, result, expected = minimal_negative_mock(kind)
            metrics = nav.check_result(request, result, expected)
            self.assertEqual(metrics[0]["kind"], kind)
            self.assertEqual(metrics[0]["id"], "MOCK_NOT_TRANSPORT_EVIDENCE")
            self.assertEqual(metrics[0]["position_max_mm"], 0)
            result["tracks"]["host"][0]["status"] = 0
            with self.assertRaisesRegex(ValueError, "terminal status"):
                nav.check_result(request, result, expected)

    def test_count_tag_audit_margin_and_final_rejections(self):
        mutations = {"count": 2, "completed_tag": 0, "crossings": 1, "path_mm": 2, "field_evaluations": -1, "substep_calls": 2, "min_margin_mm": -0.001, "status": 6}
        for key, value in mutations.items():
            request, result, expected = minimal_negative_mock()
            result["tracks"]["host"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                nav.check_result(request, result, expected)
        request, result, expected = minimal_negative_mock()
        result["tracks"]["host"][0]["final"]["position_mm"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "final mismatch"):
            nav.check_result(request, result, expected)

    def test_step_invariants_rejected(self):
        mutations = {"completed_tag": 0, "distance_mm": -1, "requested_mm": 2, "cumulative_path_mm": 0.9, "momentum_mev_c": 5.01, "time_ns": float("nan"), "field_evaluations": -1, "substep_calls": True, "looping": True, "geometry_failed": True, "boundary": True}
        for key, value in mutations.items():
            request, result, expected = minimal_negative_mock()
            result["tracks"]["host"][0]["steps"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                nav.check_result(request, result, expected)
        request, result, expected = minimal_negative_mock()
        output = result["tracks"]["host"][0]
        output["steps"][0]["after"]["position_mm"][0] += 1e-9
        output["final"] = copy.deepcopy(output["steps"][0]["after"])
        with self.assertRaisesRegex(ValueError, "hidden boundary"):
            nav.check_result(request, result, expected)

    def test_independent_endpoint_and_input_checks(self):
        request, result, expected = minimal_negative_mock()
        output = result["tracks"]["host"][0]
        for key in ("arrival", "after"):
            output["steps"][0][key]["position_mm"] = [-9.1, 0, 0]
        output["final"] = copy.deepcopy(output["steps"][0]["after"])
        with self.assertRaisesRegex(ValueError, "independent path/endpoint"):
            nav.check_result(request, result, expected)
        request, result, expected = minimal_negative_mock()
        result["input"]["tracks"][0]["delta_chord_mm"] *= 2
        with self.assertRaisesRegex(ValueError, "exact input"):
            nav.check_result(request, result, expected)
        request, result, expected = minimal_negative_mock()
        expected["budgets"]["position_mm"] *= 2
        with self.assertRaisesRegex(ValueError, "fitted"):
            nav.check_result(request, result, expected)

    def test_missing_loop_and_spurious_guard_transport_rejected(self):
        request, result, expected = minimal_negative_mock("outer_limit")
        result["tracks"]["host"][0]["steps"][0]["looping"] = False
        with self.assertRaisesRegex(ValueError, "did not exhaust"):
            nav.check_result(request, result, expected)
        expected["kind"] = "guard"
        with self.assertRaisesRegex(ValueError, "process failure"):
            nav.check_result(request, result, expected)

    def test_unused_field_margin_and_device_sync(self):
        request, result, expected = minimal_negative_mock()
        output = result["tracks"]["host"][0]
        output["field_evaluations"] = output["steps"][0]["field_evaluations"] = 0
        output["min_margin_mm"] = None
        nav.check_result(request, result, expected)
        output["min_margin_mm"] = 0
        with self.assertRaisesRegex(ValueError, "must be null"):
            nav.check_result(request, result, expected)
        request, result, expected = minimal_negative_mock()
        result["backend"] = "device"
        result["tracks"]["device"] = result["tracks"].pop("host")
        with self.assertRaisesRegex(ValueError, "synchronized"):
            nav.check_result(request, result, expected)

    def test_guard_manifest_requires_nonzero_and_specific_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _, expected = minimal_negative_mock()
            expected["kind"] = "guard"
            expected["tracks"] = []
            expected["process_marker"] = "M2FS_NAV_MAP_GUARD"
            case = {"id": "guard_mock", "kind": "guard", "request": "guard.request.json", "expected": "guard.expected.json", "result": "guard.result.json"}
            for filename, value in ((case["request"], request), (case["expected"], expected), ("fixtures.json", {"format": nav.INDEX_FORMAT, "cases": [case]})):
                (root / filename).write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "process failure evidence"):
                nav.check(root)
            run = {"id": case["id"], "request": case["request"], "result": case["result"], "returncode": 1, "backend": "host", "sanitizer_clean": False, "stderr": "guard.stderr.txt"}
            (root / "run-manifest.json").write_text(json.dumps({"cases": [run]}))
            (root / run["stderr"]).write_text("MOCK TEST TEXT ONLY: unrelated startup failure")
            with self.assertRaisesRegex(ValueError, "guard diagnostic"):
                nav.check(root)
            (root / run["stderr"]).write_text("MOCK TEST TEXT ONLY: M2FS_NAV_MAP_GUARD")
            report = nav.check(root)
            self.assertEqual(report["metrics"][0]["kind"], "guard")
            run["returncode"] = 0
            (root / "run-manifest.json").write_text(json.dumps({"cases": [run]}))
            with self.assertRaisesRegex(ValueError, "process failure evidence"):
                nav.check(root)

    def test_sanitizer_log_presence_hash_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "sanitizer.txt"
            clean = "MOCK TEST ONLY: COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
            log.write_text(clean)
            run = dict(sanitizer_log=log.name, sanitizer_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                       sanitizer_clean=True)
            nav.check_sanitizer(root, run)
            log.write_text(clean + "edited")
            with self.assertRaisesRegex(ValueError, "changed sanitizer"):
                nav.check_sanitizer(root, run)
            log.write_text(clean.replace("0 errors", "1 errors"))
            run["sanitizer_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "not clean"):
                nav.check_sanitizer(root, run)
            log.unlink()
            with self.assertRaises(FileNotFoundError):
                nav.check_sanitizer(root, run)

    def test_native_runner_manifest_hashes_and_dual_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            index = nav.prepare(root)
            case = next(case for case in index["cases"] if case["kind"] == "guard")
            (root / "fixtures.json").write_text(json.dumps({"format": nav.INDEX_FORMAT, "cases": [case]}))
            manifest = {"format": "m2fs-navigation-run-v1", "sanitizer": False, "fixture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}, "runs": []}
            for backend in ("host", "device"):
                name = "guard-" + backend
                log = root / (name + ".log")
                log.write_text("MOCK PROCESS LOG ONLY: M2FS_NAV_MAP_GUARD")
                manifest["runs"].append({"name": name, "request": case["request"], "result": name + ".result.json", "backend": backend, "returncode": 1, "accepted_process": True, "negative_guard": True, "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(), "result_sha256": None, "sanitizer_clean": None})
            (root / "run-manifest.json").write_text(json.dumps(manifest))
            report = nav.check(root)
            self.assertEqual([metric["backend"] for metric in report["metrics"]], ["host", "device"])
            self.assertTrue(report["process_manifest_checked"])
            partial = nav.check(root, ["guard"])
            self.assertFalse(partial["complete_suite"])
            self.assertEqual(partial["checked_cases"], ["guard"])
            with self.assertRaisesRegex(ValueError, "selected cases"):
                nav.check(root, ["unknown"])
            omitted = next(entry for entry in index["cases"] if entry["id"] == "outer_limit")
            (root / "fixtures.json").write_text(json.dumps({"format": nav.INDEX_FORMAT, "cases": [case, omitted]}))
            manifest["fixture_sha256"]["fixtures.json"] = hashlib.sha256((root / "fixtures.json").read_bytes()).hexdigest()
            manifest["runs"].append({"request": omitted["request"], "accepted_process": False})
            (root / "run-manifest.json").write_text(json.dumps(manifest))
            self.assertFalse(nav.check(root, ["guard"])["complete_suite"])
            with self.assertRaisesRegex(ValueError, "runner rejected process"):
                nav.check(root)
            manifest["runs"].pop()
            (root / "fixtures.json").write_text(json.dumps({"format": nav.INDEX_FORMAT, "cases": [case]}))
            manifest["fixture_sha256"]["fixtures.json"] = hashlib.sha256((root / "fixtures.json").read_bytes()).hexdigest()
            manifest["runs"].pop()
            (root / "run-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "separate-backend"):
                nav.check(root)
            with (root / case["request"]).open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "changed fixture"):
                nav.check(root)


if __name__ == "__main__":
    unittest.main()
