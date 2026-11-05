"""Grouping checker must not depend on lossy saved associations."""
from test_controls import File, load_script
from test_hit_links import Tree


def checker(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "check_hit_weights", load_script("check_hit_weights"))
    return load_script("check_hit_grouping")


def test_grouping_uses_all_source_reco_digis_not_duplicate_links(monkeypatch):
    script = checker(monkeypatch)
    monkeypatch.setattr(script.uproot, "open", lambda _: File(Tree(duplicate=True)))
    result = script.check("unused", "EventNtuple/ntuple", 2, n_crystals=8, time4merge=6.)
    assert result["certified_scope"] is True
    assert result["layout_equal"] is True
    assert result["equal_time_hits"] == 1
    assert result["nonidentical_equal_time_pairs"] == 0
    assert result["stock_hits"] == result["predicted_hits"] == 1


def test_wrong_window_is_not_certified(monkeypatch):
    import awkward as ak
    script = checker(monkeypatch)
    tree = Tree()
    reco = tree.branches["calorecodigis"].data
    tree.branches["calorecodigis"].data = ak.with_field(reco, ak.Array([[1., 12.], []]), "time_")
    monkeypatch.setattr(script.uproot, "open", lambda _: File(tree))
    result = script.check("unused", "EventNtuple/ntuple", 2, n_crystals=8, time4merge=6.)
    assert result["certified_scope"] is False
    assert result["layout_equal"] is False


def test_nonidentical_equal_times_refused_even_with_matching_float_output(monkeypatch):
    import awkward as ak
    script = checker(monkeypatch)
    tree = Tree()
    reco = tree.branches["calorecodigis"].data
    tree.branches["calorecodigis"].data = ak.with_field(reco, ak.Array([[5., 7.], []]), "eDep_")
    monkeypatch.setattr(script.uproot, "open", lambda _: File(tree))
    result = script.check("unused", "EventNtuple/ntuple", 2, n_crystals=8, time4merge=6.)
    assert result["layout_equal"] is True
    assert all(item["float32_equal"] for item in result["quantities"].values())
    assert result["nonidentical_equal_time_pairs"] == 1
    assert result["certified_scope"] is False
