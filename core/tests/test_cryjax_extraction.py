"""The CRY pieces extracted to cryjax: the image's shim source stays the cryjax source, and the old paths still work."""
import hashlib
from importlib import resources
from pathlib import Path


def test_shim_source_matches_cryjax():
    here = Path(__file__).resolve().parents[1] / "native/cry/cry_shim.cc"
    packaged = resources.files("cryjax").joinpath("native/cry_shim.cc").read_bytes()
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(packaged).hexdigest(), \
        "core/native/cry/cry_shim.cc drifted from cryjax/native/cry_shim.cc: change it in cryjax and copy it here"


def test_old_import_paths_and_mu2e_defaults(monkeypatch):
    from mu2efastsim import cry_constants, cry_native, cry_tables
    import cryjax.tables
    assert cry_tables.SEA_LEVEL_SHA256 == cry_constants.SEA_LEVEL_SHA256 == cryjax.tables.SEA_LEVEL_SHA256
    assert cry_native.CryBatch is __import__("cryjax.native", fromlist=["CryBatch"]).CryBatch
    seen = {}
    monkeypatch.delenv("CRYDATAPATH", raising=False)
    monkeypatch.delenv("MU2E_CRY_LIBRARY", raising=False)
    monkeypatch.setattr(cryjax.native.CrySource, "__init__", lambda self, **kw: seen.update(kw))
    cry_native.CrySource(seed=3)
    assert seen == {"data": "/opt/cry/data", "library": "/opt/cry/lib/libmu2ecry.so", "seed": 3}
