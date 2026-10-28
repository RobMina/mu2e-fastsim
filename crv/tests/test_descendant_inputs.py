"""Reject mismatched/overlapping populations and silently rebaselined assets."""
import pandas as pd
import pytest

from valhist import descendant_inputs as I


def frame(events, subrun=1):
    return pd.DataFrame({"run": 1400, "subrun": subrun, "event": events})


@pytest.mark.parametrize("spec", ["11,11", "11-12,12", "0", "12-11"])
def test_shards_reject_duplicates_and_empty(spec):
    with pytest.raises(ValueError):
        I.unique_shards(spec)


def test_population_checks_every_reference_identity_and_overlap():
    gen = frame([1, 2, 3])
    donors = pd.MultiIndex.from_frame(frame([1], 99))
    seen = set()
    assert len(I.population(gen, frame([2]), frame([1, 3]), donors, seen)) == 3
    with pytest.raises(ValueError, match="overlapping"):
        I.population(gen, frame([2]), frame([1]), donors, seen)
    with pytest.raises(ValueError, match="reference outside"):
        I.population(gen, frame([2]), frame([4]), donors, set())
    with pytest.raises(ValueError, match="overlaps training"):
        I.population(gen, frame([2]), frame([1]), pd.MultiIndex.from_frame(frame([2])), set())


def test_pilot_shard_guard(monkeypatch):
    monkeypatch.setattr(I, "tree_inventory", lambda *a: {"unique_events": 10, "event_keys_sha256": "shard12"})
    with pytest.raises(ValueError, match="requested shard"):
        I.bind_pilot({"generated": {"unique_events": 10, "event_keys_sha256": "shard11"}}, "path")


def test_external_assets_are_verified_not_rehashed(tmp_path):
    p = tmp_path / "asset"
    p.write_text("old")
    old = {"asset_paths": [str(p)], "files": {str(p): I.digest(p)}}
    assert I.external_assets(old) == old["files"]
    p.write_text("changed")
    with pytest.raises(ValueError, match="asset changed"):
        I.external_assets(old)


def test_report_binding_rejects_different_cache():
    with pytest.raises(ValueError, match="cache"):
        I.bind_reports({"cache": "A", "cache_sha256": "x"}, {"cache": "B", "cache_sha256": "x"}, {})
