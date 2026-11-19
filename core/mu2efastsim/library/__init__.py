"""Mu2e view of the shared library core, which now lives in the experiment-neutral package hep-simlib
(import name hepsimlib, pinned in pyproject.toml; spec: core/docs/libraries.md and hep-simlib's docs/libraries.md).

Importing this package installs the Mu2e profile (MU2E below) and makes ``mu2efastsim.library.<module>`` the very same
module object as ``hepsimlib.library.<module>``, so every existing import path, attribute and module-level state is
unchanged:

    hashing     canonical JSON, file sha256, array content hash, short digests
    axes        Axis (key, unit, frame, edges, out-of-domain policy), Grid (row-major cells), NumPy/JAX bin lookup
    domain      OutOfDomain (loud refusal), DomainCounter (declared fallbacks and clips, counted)
    manifest    the manifest schema (provenance, split, seeds, dtype and RNG conventions, validation status)
    store       .npz library files with an embedded or sidecar manifest; load() checks everything it can
    registry    shipped names, registry JSON, release assets and the download cache (sha256 checked each time)
    dtypes      the float32/float64 compute policy and device copies
    provenance  process-wide record of the libraries a run loaded, pinned or unpinned
    fidelity    the validation vocabulary: status levels, metrics, criteria declared before the held-out read,
                validation records, and the bench performance entry

The Mu2e profile keeps the settings these modules had before the extraction: the subsystem list, MU2EFS_DEVICE_FLOAT
(float32 when unset), MU2EFS_LIBRARY_CACHE and ~/.cache/mu2efastsim/libraries, the mu2e-fastsim release URL, and the
component bank names and reason texts (secondaries.component.*).
"""
import importlib
import sys

from hepsimlib import profile as _profile

MU2E = _profile.Profile(
    name="mu2e",
    subsystems=("core", "crv", "calo", "tracker", "flash", "secondaries", "beam", "trigger", "mixing"),
    dtype_env="MU2EFS_DEVICE_FLOAT", default_dtype="float32",
    cache_env="MU2EFS_LIBRARY_CACHE", cache_dir="~/.cache/mu2efastsim/libraries",
    release_url="https://github.com/RobMina/mu2e-fastsim/releases/download/{tag}/{file}",
    bank_subsystem="secondaries", bank_prefix="secondaries.component.",
    bank_reasons=dict(source="the capture products the bank was built from are not given",
                      generator="physics-list, material and cut hashes are not recorded yet (capture P2b)",
                      release="not given to save_library()", seeds="not given to save_library()",
                      split="not given to save_library()", config_hash="no build configuration manifest"),
    bank_validation_reason="a complete manifest is necessary, not sufficient (docs/component-training.md); the energy "
                           "accounting stays unrecorded until the bank carries a capture_accounting report hash")
_profile.configure(MU2E)

MODULES = ("axes", "domain", "dtypes", "fidelity", "hashing", "manifest", "provenance", "registry", "store")
for _name in MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"hepsimlib.library.{_name}")

from hepsimlib.library import (axes, domain, dtypes, fidelity, hashing, manifest, provenance,  # noqa: E402,F401
                               registry, store)
from hepsimlib.library.axes import Axis, Grid  # noqa: E402,F401
from hepsimlib.library.domain import DomainCounter, OutOfDomain  # noqa: E402,F401
from hepsimlib.library.registry import Registry  # noqa: E402,F401
from hepsimlib.library.store import Library, LibraryError, load, save  # noqa: E402,F401
