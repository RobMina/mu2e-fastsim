"""Combined-replay guards; optional saved-run tests never fabricate transport success."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parents[1] / "celeritas_adapter/navigation_probe"


def load(name):
    spec = importlib.util.spec_from_file_location("navigation_report_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = load("fixtures")
with patch.dict(sys.modules, {"fixtures": fixtures}):
    reporter = load("report")


class CombinedReplayTest(unittest.TestCase):
    def test_build_identity_must_match_and_require_memcheck(self):
        first = {key: "synthetic identity only" for key in reporter.IDENTITY_KEYS}
        first.update(format="m2fs-navigation-run-v1", sanitizer=True)
        self.assertEqual(set(reporter.verify_identity([first, copy.deepcopy(first)])), set(reporter.IDENTITY_KEYS))
        for key in (*reporter.IDENTITY_KEYS, "sanitizer", "format"):
            second = copy.deepcopy(first)
            second[key] = False if key == "sanitizer" else "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporter.verify_identity([first, second])

    def test_only_outer_trial_cap_may_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, remaining = Path(temporary) / "first", Path(temporary) / "remaining"
            fixtures.prepare(first, ["outer_limit"])
            fixtures.prepare(remaining, ["outer_limit"])
            request = reporter.read(first / "outer_limit.request.json")
            request["tracks"][0]["max_step_mm"] = 100.0
            (first / "outer_limit.request.json").write_text(json.dumps(request))
            expected = reporter.read(first / "outer_limit.expected.json")
            expected["request_sha256"] = fixtures.digest(request)
            (first / "outer_limit.expected.json").write_text(json.dumps(expected))
            reporter.verify_outer_revision(first, remaining)
            request["tracks"][0]["epsilon_rel_max"] *= 10
            (first / "outer_limit.request.json").write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, "more than trial cap"):
                reporter.verify_outer_revision(first, remaining)
            request["tracks"][0]["epsilon_rel_max"] /= 10
            (first / "outer_limit.request.json").write_text(json.dumps(request))
            expected["budgets"]["position_mm"] *= 2
            (first / "outer_limit.expected.json").write_text(json.dumps(expected))
            with self.assertRaisesRegex(ValueError, "oracle/budgets changed"):
                reporter.verify_outer_revision(first, remaining)

    def test_revised_fixture_comparison_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference, saved = Path(temporary) / "reference", Path(temporary) / "saved"
            index = fixtures.prepare(reference, ["zero"])
            fixtures.prepare(saved, ["zero"])
            reporter.verify_fixture(saved, reference, index["cases"][0])
            with (saved / "zero.gdml").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "not the revised fixture"):
                reporter.verify_fixture(saved, reference, index["cases"][0])

    @unittest.skipUnless(os.environ.get("M2FS_NAV_ORIGINAL_RESULTS") and os.environ.get("M2FS_NAV_REMAINING_RESULTS"),
                         "set real saved navigation results for combined replay")
    def test_real_saved_replay_preserves_failed_original(self):
        result = reporter.report(Path(os.environ["M2FS_NAV_ORIGINAL_RESULTS"]), Path(os.environ["M2FS_NAV_REMAINING_RESULTS"]))
        self.assertTrue(result["revised_suite_pass"])
        self.assertFalse(result["single_run_pass"])
        self.assertFalse(result["original_failure"]["accepted_process"])
        self.assertEqual(result["counts"]["fixture_cases"], 8)
        self.assertEqual(result["counts"]["positive_track_backend_checks"], 42)
        self.assertEqual(result["counts"]["exhaustion_track_backend_checks"], 4)
        self.assertEqual(result["counts"]["guard_process_checks"], 2)
        self.assertEqual(result["counts"]["sanitizer_clean_cases"], 7)
        self.assertTrue(all(not run["replay"]["complete_suite"] for run in result["runs"]))


if __name__ == "__main__":
    unittest.main()
