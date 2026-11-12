"""The mixing chains load the CRV optical assets through the run configuration's pinned asset set
(core/docs/libraries.md, "Pinning in run configurations"): every lookup-table and photon-map load of mixing/ sits
inside ``with lookup.pinned(...)``. The loads themselves are unchanged, so the outputs are the same arrays."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "mixing"
LOADERS = {"load_tables", "load_device_tables", "photon_map"}


def _loader_calls(tree):
    """(call node, inside a `with lookup.pinned(...)` block) for every optical-asset loader call."""
    out = []

    def visit(node, pinned):
        if isinstance(node, ast.With):
            pinned = pinned or any(isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Attribute)
                                   and i.context_expr.func.attr == "pinned" for i in node.items)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in LOADERS:
            out.append((node, pinned))
        for child in ast.iter_child_nodes(node):
            visit(child, pinned)

    visit(tree, False)
    return out


@pytest.mark.parametrize("module", ["device_physics.py", "fast_detectors.py", "stream_backend.py"])
def test_production_crv_asset_loads_are_pinned(module):
    calls = _loader_calls(ast.parse((ROOT / module).read_text()))
    assert calls, f"{module}: no optical-asset load found"
    unpinned = [ast.unparse(c) for c, pinned in calls if not pinned]
    assert not unpinned, f"{module}: loads outside lookup.pinned(...): {unpinned}"


def test_every_mixing_module_pins_its_crv_asset_loads():
    for path in sorted(ROOT.rglob("*.py")):
        unpinned = [ast.unparse(c) for c, pinned in _loader_calls(ast.parse(path.read_text())) if not pinned]
        assert not unpinned, f"{path.name}: loads outside lookup.pinned(...): {unpinned}"


def test_the_run1a_configuration_has_the_pinned_asset_set():
    from crvfastsim import runconfig
    pins = runconfig.get("run1a_v01").asset_pin_map()
    assert set(pins) >= {"LookupTable_6000_0", "photonMap.root"}
    assert all(len(p["content_hash"]) == 64 for p in pins.values())
