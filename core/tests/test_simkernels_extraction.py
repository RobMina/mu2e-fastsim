"""The physics kernels moved to hep-simkernels: the Mu2e import paths are the hepsimkernels modules themselves, the
MU2EFS_ variable names are kept, and Mu2e's materials, production cuts, field-map scale and device-loop provenance
are installed through hepsimkernels.settings."""
import subprocess
import sys
from pathlib import Path

import hepsimkernels
import hepsimkernels.settings as kernel_settings
from mu2efastsim import bfield_scale, materials

ROOT = Path(__file__).resolve().parents[1]
MOVED = ("eloss", "eloss_jax", "eioni", "emsc", "ebrem", "eannihil", "muradiative", "muradiative_jax", "eslab",
         "eslab_jax", "slab_jax", "transport", "world", "ehelix", "crossings", "precision", "determinism")
FORMER_ORDER = ("G4_POLYSTYRENE", "G4_Al", "G4_AIR", "CONCRETE_MARS", "MBOverburden", "MildSteel", "ShieldingConcrete",
                "G4_Cu", "G4_Si", "Electronics2", "CuW1090", "Al6061", "Al7075", "BronzeC938", "TrackerManifold", "G10",
                "Electronics", "G4_POLYETHYLENE", "G4_SILICON_DIOXIDE", "G4_MYLAR", "G4_W", "G4_Au", "G4_Pb",
                "AluminumHoneycomb", "CarbonFiber", "CalCableRunOuter", "CalCableRunFiber", "TrkCableRunFiber",
                "TrkCableRunOuter", "G4_CESIUM_IODIDE", "Polyethylene092", "Polyetheretherketone", "Lyso_01",
                "StainlessSteel")


def test_import_paths_are_the_hepsimkernels_modules():
    import importlib
    for name in MOVED:
        assert importlib.import_module(f"mu2efastsim.{name}") is importlib.import_module(f"hepsimkernels.{name}")
    from mu2efastsim.geometry import annulus_jax
    assert annulus_jax is importlib.import_module("hepsimkernels.geometry.annulus_jax")


def test_settings_keep_the_mu2e_names_and_hooks():
    assert kernel_settings.PREFIX == "MU2EFS" and kernel_settings.name("DEVICE_FLOAT") == "MU2EFS_DEVICE_FLOAT"
    assert kernel_settings.MAP_SCALE is bfield_scale.factor
    assert "mu2efastsim.materials" in kernel_settings.MATERIAL_MODULES
    from mu2efastsim import precision
    p = precision.provenance()
    assert list(p)[:4] == ["device_float", "deterministic", "device_loops", "graph_concurrency"]


def test_materials_registry_keeps_the_former_order_and_cuts():
    import hepsimkernels.materials as core
    assert materials.BY_NAME is core.BY_NAME and materials.get is core.get
    assert tuple(core.BY_NAME) == FORMER_ORDER
    assert materials.POLYSTYRENE is core.POLYSTYRENE and core.POLYSTYRENE.cuts_e_MeV == {"crv": 0.0872, "tagger": 2.2296}
    assert core.get("CONCRETE_MARS") is materials.CONCRETE_MARS


def test_mu2e_materials_registered_without_importing_the_wrapper_first():
    code = ("import mu2efastsim.transport, sys; import hepsimkernels.materials as m; "
            "assert 'CONCRETE_MARS' in m.BY_NAME and m.get('G4_POLYSTYRENE').cuts_e_MeV['crv'] == 0.0872; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_importing_the_package_loads_no_numpy_or_precision():
    code = "import mu2efastsim, sys; print('numpy' in sys.modules, 'hepsimkernels.precision' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0 and r.stdout.split() == ["False", "False"], (r.stdout, r.stderr)
    assert hepsimkernels.__version__
