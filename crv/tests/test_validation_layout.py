"""crvfastsim.validation: the moved modules, their one-release aliases and the package layering."""
import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = ("v0check", "v1compare", "v2compare", "v4photons", "v4sipm", "transport_validation")
VALHIST = ("compare", "coverage", "pilot", "paired_rates", "paired_shapes", "event_shapes", "reco_detail")


@pytest.mark.parametrize("old,name", [("crvfastsim.analyses", n) for n in ANALYSES] + [("valhist", n) for n in VALHIST])
def test_old_paths_alias_the_moved_module(old, name):
    moved = importlib.import_module(f"crvfastsim.validation.{name}")
    assert importlib.import_module(f"{old}.{name}") is moved


def test_old_module_path_still_runs_as_a_script():
    r = subprocess.run([sys.executable, "-m", "valhist.compare", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "merge" in r.stdout


def test_crvfastsim_never_imports_valhist():
    bad = []
    for path in (ROOT / "crvfastsim").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            bad += [f"{path.relative_to(ROOT)}: {n}" for n in names if n == "valhist" or n.startswith("valhist.")]
    assert not bad, bad
