"""The scorer, the warm-rate timer, the approximation registry and the robust statistics moved to hep-simbench:
the Mu2e import paths are the hepsimbench modules themselves (the timer keeps Mu2e's GPU launch options as an
extension), and every command line and report field is as before."""
import json
import subprocess
import sys
from pathlib import Path

import hepsimbench.approximations
import hepsimbench.bench
import hepsimbench.stats
from mu2efastsim import approximations, bench, stats

ROOT = Path(__file__).resolve().parents[2]


def test_import_paths_are_the_hepsimbench_modules():
    assert approximations is hepsimbench.approximations and stats is hepsimbench.stats
    assert approximations.Approximation is hepsimbench.approximations.Approximation
    for name in ("run", "summarise", "time_passes", "fresh_inputs", "FreshInputError", "device_label",
                 "parse_nvidia_smi", "resolve", "format_report", "accepts_seed", "device_sync"):
        assert getattr(bench, name) is getattr(hepsimbench.bench, name)


def test_benchmark_paths_are_the_hepsimbench_modules():
    sys.path.insert(0, str(ROOT))
    try:
        import hepsimbench.from_root
        import hepsimbench.score
        from benchmarks.fastsim_score import from_root, score
        assert score is hepsimbench.score and from_root is hepsimbench.from_root
    finally:
        sys.path.remove(str(ROOT))


def test_bench_keeps_the_mu2e_launch_options_and_report_fields(tmp_path):
    out = tmp_path / "bench.json"
    rc = bench.main(["builtins:len", "--events", "3", "--prepare", "builtins:range", "--device-loops", "off",
                     "--graph-concurrency", "off", "--device-label", "t", "--out", str(out)])
    assert rc == 0
    r = json.loads(out.read_text())
    keys = list(r)
    assert keys[keys.index("entry") + 1:keys.index("entry") + 3] == ["device_loops", "graph_concurrency"]
    assert r["command"].startswith("python3 -m mu2efastsim.bench builtins:len")


def test_module_commands_still_run(tmp_path):
    p = subprocess.run([sys.executable, "-m", "mu2efastsim.bench", "--help"], capture_output=True, text=True)
    assert p.returncode == 0 and "--device-loops" in p.stdout and "--approx-preset" in p.stdout
    p = subprocess.run([sys.executable, "-m", "mu2efastsim.approximations", "--help"], capture_output=True, text=True)
    assert p.returncode == 0
