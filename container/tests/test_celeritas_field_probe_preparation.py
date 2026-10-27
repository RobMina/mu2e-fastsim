"""Offline preparation/runner guards; no compiler or GPU execution."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parents[1] / "celeritas_adapter/field_probe"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_probe")
runner = load("run_probe")


class ProbePreparationTest(unittest.TestCase):
    def test_preparation_wrong_pin_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter-manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "pinned"):
                prepare.prepare(adapter, root / "output")
            self.assertFalse((root / "output").exists())

    def test_modified_parser_cannot_be_staged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter-manifest.json").write_text(json.dumps({"prepared_sha256": {"CartMapIO.hh": "expected"}}))
            with mock.patch.object(prepare, "digest", side_effect=["774e93dc8e2f8aec55b0e76947cb237c72300d5e1d122ccc89ca08b57fe28933", "changed"]):
                with self.assertRaisesRegex(ValueError, "parser changed"):
                    prepare.prepare(root, root / "output")
            self.assertFalse((root / "output").exists())

    def test_linkage_requires_h5_family(self):
        prefix = Path("/h5/install")
        good = "\n".join(f"lib{name}.so => /h5/install/lib64/lib{name}.so (0x1)"
                         for name in ("celeritas", "geocel", "corecel"))
        self.assertEqual(len(runner.resolve_libraries(good, prefix)), 3)
        for bad in (good.replace("/h5/install/lib64/libceleritas", "/opt/view/lib64/libceleritas"),
                    good.replace("/h5/install/", "/h5/install-old/"),
                    good + "\nlibcuda.so => not found", good.splitlines()[0]):
            with self.assertRaises(ValueError):
                runner.resolve_libraries(bad, prefix)

    def test_no_build_in_prepare_or_runner(self):
        for name in ("prepare_probe.py", "run_probe.py"):
            text = (HERE / name).read_text()
            self.assertNotIn("cmake --build", text)
            self.assertNotIn('subprocess.run(["cmake"', text)
        source = (HERE / "ProbeAlgorithms.hh").read_text()
        self.assertIn("celeritas::DormandPrinceIntegrator", source)
        self.assertIn("celeritas::FieldSubstepper", source)
        self.assertIn("numerical_failures", source)
        self.assertLess(source.index("if (margin < 0)"), source.index("auto result = field_(pos)"))

    def test_numerical_guard_and_work_bounds_present(self):
        text = (HERE / "probe.cc").read_text()
        self.assertIn("length_mm >= 1e-4", text)
        self.assertIn("result.steps > 0", text)
        kernel = (HERE / "ProbeKernels.cu").read_text()
        self.assertIn("cudaMemset", kernel)
        self.assertIn("cudaGetLastError", kernel)
        self.assertIn("cudaDeviceSynchronize", kernel)


if __name__ == "__main__":
    unittest.main()
