"""The library core and the component banks moved to hep-simlib: the Mu2e import paths are the hepsimlib modules
themselves, and the Mu2e profile keeps every setting the modules had before (subsystems, MU2EFS_DEVICE_FLOAT with a
float32 default, the cache, the release URL, bank names and texts)."""
import subprocess
import sys
from pathlib import Path

import numpy as np

import hepsimlib.component
import hepsimlib.library as core
from hepsimlib import profile
from mu2efastsim import component_response, library
from mu2efastsim.library import dtypes, manifest, registry


def test_import_paths_are_the_hepsimlib_modules():
    for name in library.MODULES:
        assert sys.modules[f"mu2efastsim.library.{name}"] is getattr(core, name) is getattr(library, name)
    assert component_response is hepsimlib.component
    assert library.store.load is core.store.load and library.Axis is core.Axis


def test_the_mu2e_profile_is_installed_and_keeps_the_settings(monkeypatch):
    assert profile.current() is library.MU2E
    assert manifest.SUBSYSTEMS == ("core", "crv", "calo", "tracker", "flash", "secondaries", "beam", "trigger", "mixing")
    monkeypatch.delenv("MU2EFS_DEVICE_FLOAT", raising=False)
    assert dtypes.compute_dtype() == np.float32
    monkeypatch.setenv("MU2EFS_DEVICE_FLOAT", "float64")
    assert dtypes.compute_dtype() == np.float64
    monkeypatch.delenv("MU2EFS_LIBRARY_CACHE", raising=False)
    assert registry.CACHE_ENV == "MU2EFS_LIBRARY_CACHE"
    assert registry.default_cache() == Path.home() / ".cache" / "mu2efastsim" / "libraries"
    assert registry.release_url("tracker.nrow.run1a", 2, "f.npz") == \
        "https://github.com/RobMina/mu2e-fastsim/releases/download/lib-tracker.nrow.run1a-v2/f.npz"
    assert component_response.BANK_PREFIX == "secondaries.component."
    assert library.MU2E.bank_reasons["generator"].endswith("(capture P2b)")


def test_the_library_command_line_keeps_its_name():
    out = subprocess.run([sys.executable, "-m", "mu2efastsim.library", "--help"], capture_output=True, text=True,
                         check=True).stdout
    assert out.startswith("usage: python3 -m mu2efastsim.library")
