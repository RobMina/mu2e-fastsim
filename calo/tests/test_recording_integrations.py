"""Collect review-only manifest/inventory tests in the standard calorimeter suite."""
from pathlib import Path
import importlib.util


HERE = Path(__file__).resolve().parents[1] / "recording"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = _load("manifest_source_checks", HERE / "manifest/tests/test_manifest_source.py")
for name, function in vars(manifest).items():
    if name.startswith("test_"):
        globals()["test_manifest_" + name[5:]] = function

inventory = _load("inventory_source_checks", HERE / "inventory/tests/test_inventory.py")
for name in ("test_source_invariants", "test_guard_deletion_mutations_are_detected",
             "test_manifest_patch_footprint_and_hook_order"):
    globals()["test_inventory_" + name[5:]] = getattr(inventory, name)
