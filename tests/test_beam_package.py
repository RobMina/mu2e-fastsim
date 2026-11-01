"""beam/pyproject.toml: the package, its data files and every third-party import are declared."""
import ast
import sys
import tomllib
from pathlib import Path

BEAM = Path(__file__).resolve().parents[1] / "beam"
PROJECT = tomllib.loads((BEAM / "pyproject.toml").read_text())


def test_package_data_exists():
    for package, globs in PROJECT["tool"]["setuptools"]["package-data"].items():
        for pattern in globs:
            assert list((BEAM / package.replace(".", "/")).glob(pattern)), (package, pattern)


def test_every_third_party_import_is_declared():
    declared = {d.split("[")[0].split(">")[0].split("=")[0].strip() for d in PROJECT["project"]["dependencies"]}
    declared |= {d for extra in PROJECT["project"]["optional-dependencies"].values() for d in extra}
    seen = set()
    for path in (BEAM / "beamfastsim").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                seen |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                seen.add(node.module.split(".")[0])
    third_party = seen - set(sys.stdlib_module_names) - {"beamfastsim"}
    assert third_party <= declared, sorted(third_party - declared)


def test_console_script_target_exists():
    module, func = PROJECT["project"]["scripts"]["beamfastsim"].split(":")
    tree = ast.parse((BEAM / (module.replace(".", "/") + ".py")).read_text())
    assert func in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
