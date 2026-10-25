"""Allocation and failure guards for the load-only probe, using fake executables."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "probe_celeritas.sh"


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.image = self.root / "image with spaces.sif"
        self.image.touch()
        self.out = self.root / "results"
        self.calls = self.root / "calls"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        SLURM_JOB_ID="123", SLURM_STEP_ID="0", CALLS=str(self.calls))
        self.fake("nvidia-smi", 'echo "NVIDIA A100-SXM4-80GB, GPU-fake, 999"')
        self.fake("stat", 'echo "image_bytes=0 image_mtime=mock"')
        self.fake("apptainer", 'printf "%s\\n" "$*" >> "$CALLS"\necho mock')

    def fake(self, name, body):
        path = self.bin / name
        path.write_text(f"#!/bin/bash\n{body}\n")
        path.chmod(0o755)

    def run_probe(self):
        return subprocess.run(["bash", str(SCRIPT), str(self.image), str(self.out)],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def test_no_allocation(self):
        self.env.pop("SLURM_JOB_ID")
        self.assertEqual(self.run_probe().returncode, 2)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.calls.exists())

    def test_no_step(self):
        self.env.pop("SLURM_STEP_ID")
        self.assertEqual(self.run_probe().returncode, 2)
        self.assertFalse(self.calls.exists())

    def test_wrong_gpu(self):
        self.fake("nvidia-smi", 'echo "NVIDIA B200, GPU-fake, 999"')
        self.assertEqual(self.run_probe().returncode, 2)
        self.assertFalse(self.out.exists())

    def test_missing_image(self):
        self.image.unlink()
        self.assertNotEqual(self.run_probe().returncode, 0)
        self.assertFalse(self.calls.exists())

    def test_preserves_existing_output(self):
        self.out.mkdir()
        marker = self.out / "probe.log"
        marker.write_text("previous evidence")
        self.assertNotEqual(self.run_probe().returncode, 0)
        self.assertEqual(marker.read_text(), "previous evidence")
        self.assertFalse(self.calls.exists())

    def test_load_only(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.out / "exit_code.txt").read_text(), "0\n")
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 4)
        for option, call in zip(("--help", "--version", "--dump-default"), calls[1:]):
            self.assertIn("exec --cleanenv --nv", call)
            self.assertTrue(call.endswith(f"celer-g4 {option}"))

    def test_loader_failure_is_preserved(self):
        self.fake("apptainer", 'if [[ $1 == exec ]]; then echo loader-error >&2; exit 17; fi')
        self.assertEqual(self.run_probe().returncode, 17)
        self.assertEqual((self.out / "exit_code.txt").read_text(), "17\n")
        self.assertIn("loader-error", (self.out / "help.stderr.txt").read_text())
        self.assertFalse((self.out / "version.txt").exists())


if __name__ == "__main__":
    unittest.main()
