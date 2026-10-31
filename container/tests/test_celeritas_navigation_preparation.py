"""Source/staging guards only; no C++ build, GPU run or manufactured runtime success."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "celeritas_adapter/navigation_probe"


def load(name):
    spec = importlib.util.spec_from_file_location("navigation_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare")
runner = load("run_probe")


class PreparationTest(unittest.TestCase):
    def test_source_pins_match_read_only_upstream(self):
        if not os.environ.get("M2FS_CELERITAS_SOURCE"):
            self.skipTest("set M2FS_CELERITAS_SOURCE to the pinned upstream source for this optional check")
        upstream = Path(os.environ["M2FS_CELERITAS_SOURCE"])
        for name, digest in prepare.API_PINS.items():
            self.assertEqual(prepare.digest(upstream / "src" / name), digest, name)

    def test_stage_is_exclusive_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter, upstream, out = root / "adapter", root / "upstream", root / "output"
            adapter.mkdir()
            upstream.mkdir()
            for name in ("CartMapIO.cc", "CartMapIO.hh"):
                (adapter / name).write_text("synthetic parser " + name)
            manifest = adapter / "adapter-manifest.json"
            manifest.write_text(json.dumps({"prepared_sha256": {name: prepare.digest(adapter / name) for name in ("CartMapIO.cc", "CartMapIO.hh")}}))
            before = prepare.digest(manifest)
            with patch.object(prepare, "ADAPTER_PIN", before), patch.object(prepare, "API_PINS", {}):
                report = prepare.prepare(adapter, upstream, out)
                self.assertEqual(report["format"], "m2fs-navigation-source-v1")
                self.assertEqual(prepare.digest(manifest), before)
                for name, digest in report["sha256"].items():
                    self.assertEqual(prepare.digest(out / name), digest)
                with self.assertRaises(ValueError):
                    prepare.prepare(adapter, upstream, out)
                (adapter / "CartMapIO.cc").write_text("changed")
                with self.assertRaises(ValueError):
                    prepare.prepare(adapter, upstream, root / "bad")
                self.assertFalse((root / "bad").exists())

    def test_runtime_only_and_native_translation_units(self):
        wrapper = (HERE / "run_probe.sh").read_text()
        for text in (wrapper, (HERE / "run_probe.py").read_text(), (HERE / "prepare.py").read_text()):
            self.assertNotIn("cmake --build", text)
            self.assertNotIn("ninja ", text)
        cmake = (HERE / "CMakeLists.txt").read_text()
        self.assertIn("celeritas_target_link_libraries", cmake)
        self.assertIn("CUDA_SEPARABLE_COMPILATION", cmake)
        self.assertIn("Celeritas 0.6.3 EXACT", cmake)
        self.assertIn("NavKernels.cu", cmake)
        code = (HERE / "NavAlgorithms.hh").read_text()
        self.assertIn("moved.distance < rec.requested", code)
        self.assertIn("&& !moved.boundary", code)
        self.assertIn("geo.cross_boundary()", code)
        self.assertNotIn("geo.normal()", code)

    def test_vector_serialization_explicitly_converts_unit_constants(self):
        code = (HERE / "probe.cc").read_text()
        vectors = ("state.position", "geometry.bbox().lower()", "geometry.bbox().upper()",
                   "domain.lower", "domain.upper")
        for vector in vectors:
            self.assertIn(f"vector_json({vector}, static_cast<double>(millimeter))", code)
        self.assertNotRegex(code, r"vector_json\([^;\n]*,\s*millimeter\s*\)")
        self.assertIn("vector_json(state.direction, 1)", code)

    def test_expected_guard_cannot_hide_crash_timeout_or_output(self):
        marker = "M2FS_NAV_MAP_GUARD\n"
        self.assertTrue(runner.accepted_process(1, False, marker, True))
        for code, exists, log in ((0, False, marker), (124, False, marker), (1, True, marker), (1, False, "bad allocation")):
            self.assertFalse(runner.accepted_process(code, exists, log, True))
        clean = "COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
        self.assertTrue(runner.accepted_process(0, True, "", False, clean))
        for text in ("missing", clean.replace("0 errors", "1 errors"), clean + "========= ERROR: hidden"):
            self.assertFalse(runner.accepted_process(0, True, "", False, text))
        self.assertFalse(runner.accepted_process(1, True, "", False, clean))

    def test_no_mixed_library_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            (prefix / "lib").mkdir()
            rows = []
            for name in ("celeritas", "geocel", "corecel"):
                lib = prefix / "lib" / f"lib{name}.so"
                lib.write_bytes(name.encode())
                rows.append(f"lib{name}.so => {lib} (0x1)")
            text = "\n".join(rows)
            self.assertEqual(len(runner.libraries(text, prefix)), 3)
            for bad in (text.replace(str(prefix), "/opt/view"), text + "\nlibcuda.so => not found", rows[0]):
                with self.assertRaises(ValueError):
                    runner.libraries(bad, prefix)


if __name__ == "__main__":
    unittest.main()
