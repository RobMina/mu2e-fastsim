"""Tier-1b asset sets on the shared library core: a named set is pinned by its library sha256 on every load path and
recorded pinned; a custom library is recorded unpinned; the sample's placement counters map onto a DomainCounter.
The loaded library is the same object tree as before (the real Run 1A set where it is available)."""
import os

import pytest

from crvfastsim import tier1b_assets as T
from mu2efastsim.library import provenance as LP
from test_tier1b_assets import _fake_set

RUN1A_DIR = os.environ.get("CRVFS_TIER1B_RUN1A_DIR", "")


def test_named_set_is_recorded_pinned_and_custom_unpinned(tmp_path, monkeypatch):
    spec, _ = _fake_set(tmp_path, monkeypatch)
    pinned = T.Tier1bSpec(spec.name, spec.library, spec.topology, spec.templates, T.sha256(spec.library), spec.options)
    LP.clear()
    T.load(pinned, geom=None)
    T.load(T.Tier1bSpec(None, spec.library), geom=None)
    a, b = LP.loaded()
    assert a["status"] == "pinned" and a["pin_set"] == spec.name and a["content_hash"] == T.sha256(spec.library)
    assert b["status"] == "unpinned" and b["name"] == "crv.tier1b.custom" and not LP.all_pinned()


def test_the_pin_is_checked_even_without_sidecars(tmp_path, monkeypatch):
    spec, calls = _fake_set(tmp_path, monkeypatch)
    bare = T.Tier1bSpec(spec.name, spec.library, None, None, "0" * 64, spec.options)
    with pytest.raises(ValueError, match="not the pinned library"):
        T.load(bare, geom=None)
    assert calls == []                                   # refused before the library is read


def test_domain_counts_from_the_sample_info():
    info = {"desc_steps": 100, "desc_dropped": 3, "desc_clamped": 7, "other_bar": {"other": 5, "no_bar": 3}}
    rec = T.domain_counts(info).record()
    assert rec["lookups"] == 103 and rec["refused"] == 3
    assert rec["fallbacks"]["placement"]["levels"] == {"0": 88, "1": 7, "2": 5}
    c = T.domain_counts(info)
    T.domain_counts(info, c)
    assert c.lookups == 206 and c.levels["placement"]["1"] == 14


@pytest.mark.skipif(not (RUN1A_DIR and os.path.isdir(RUN1A_DIR)), reason="Run 1A tier-1b set not available")
def test_real_run1a_set_loads_the_same_library(monkeypatch):
    from crvfastsim import runconfig
    from crvfastsim.analyses.context_audit import content_fingerprint
    monkeypatch.delenv("CRVFS_PREPARED_LIBRARY_CACHE", raising=False)
    spec = T.resolve("run1a_v01", directory=RUN1A_DIR, run_config="run1a_v01")
    geom = runconfig.get("run1a_v01").geometry()
    LP.clear()
    new = T.load(spec, geom)
    assert LP.loaded()[-1]["status"] == "pinned"
    assert content_fingerprint(new) == content_fingerprint(T._load_uncached(spec, geom))   # the pre-migration path
