"""Import direction between the packages of the repository: core < detectors < workflows.

core (mu2efastsim) imports no other package of the repository; the detectors (crvfastsim and valhist, trkfastsim,
calofastsim) import core and each other; the workflows (beamfastsim, mixing, trigger) import anything. Every Python
file under a package directory counts (package code, tests, scripts), and in it every import statement (function-local
ones included) and every importlib.import_module, pytest.importorskip or __import__ call with a literal module name;
relative imports stay inside their package. Not counted: names built at run time and `python -m <package>` command
lines. The umbrella's own tests/ and container/ are not checked.

KNOWN lists today's exceptions as (file, imported package). It may only shrink: a new upward import fails, and so
does a KNOWN entry that no longer occurs (delete it from the list).
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYER = {"mu2efastsim": 0, "crvfastsim": 1, "valhist": 1, "trkfastsim": 1, "calofastsim": 1,
         "beamfastsim": 2, "mixing": 2, "trigger": 2}
DIRECTORY = {"core": 0, "crv": 1, "tracker": 1, "calo": 1, "beam": 2, "mixing": 2, "trigger": 2}
NAMES = ("core", "detector", "workflow")

KNOWN = {
    # core -> crv, beam, tracker: the endpoint kit's task registry and parameter defaults, and their tests
    ("core/mu2efastsim/endpoints/kit.py", "beamfastsim"),
    ("core/mu2efastsim/endpoints/kit.py", "crvfastsim"),
    ("core/mu2efastsim/endpoints/kit.py", "trkfastsim"),
    ("core/mu2efastsim/endpoints/parameters.py", "crvfastsim"),
    ("core/mu2efastsim/endpoints/capabilities.py", "crvfastsim"),
    ("core/tests/test_capabilities.py", "crvfastsim"),
    ("core/tests/test_crv_response_config.py", "crvfastsim"),
    ("core/tests/test_endpoint_eaf.py", "crvfastsim"),
    ("core/tests/test_endpoint_identity.py", "beamfastsim"),
    ("core/tests/test_endpoint_identity.py", "crvfastsim"),
    ("core/tests/test_kit_parameters.py", "beamfastsim"),
    ("core/tests/test_kit_parameters.py", "crvfastsim"),
    # calo -> beam: the shower hand-off reads the beam chain's outputs
    ("calo/calofastsim/showers/handoff_device.py", "beamfastsim"),
    ("calo/calofastsim/showers/reconstruct.py", "beamfastsim"),
    ("calo/calofastsim/showers/measure_chain.py", "beamfastsim"),
}


DYNAMIC = {"import_module", "importorskip", "__import__"}


def _called(node):
    """Name of the function a call node calls (import_module for importlib.import_module), or ""."""
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def imported_packages(path):
    """Top-level names of the repository's packages imported anywhere in one file (absolute imports only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif (isinstance(node, ast.Call) and _called(node) in DYNAMIC and node.args
              and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            names = [node.args[0].value]
        else:
            continue
        found.update(n.split(".")[0] for n in names if n.split(".")[0] in LAYER)
    return found


def upward_imports():
    """(file, package) for every import of a package in a higher layer than the file's directory."""
    out = set()
    for directory, layer in DIRECTORY.items():
        for path in sorted((ROOT / directory).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            for package in imported_packages(path):
                if LAYER[package] > layer:
                    out.add((relative, package))
    return out


def test_no_new_upward_imports():
    new = sorted(upward_imports() - KNOWN)
    assert not new, "imports against the direction core < detector < workflow:\n" + "\n".join(
        f"{f} ({NAMES[DIRECTORY[f.split('/')[0]]]}) imports {p} ({NAMES[LAYER[p]]})" for f, p in new)


def test_known_exceptions_still_occur():
    gone = sorted(KNOWN - upward_imports())
    assert not gone, "remove these entries from KNOWN (the import is gone):\n" + "\n".join(map(str, gone))
