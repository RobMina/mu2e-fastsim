import numpy as np
import pytest

from test_reco_combo import _sh
from trkfastsim.reco import combo, linefind
from trkfastsim.reco.hitreco import FLAG


@pytest.mark.parametrize("case", ["empty", "dead", "unselected"])
def test_panel_hits_empty_result_preserves_schema(case):
    reference = combo.panel_hits(_sh([0], [1], [10]))
    sh = _sh([0], [1], [10])
    if case == "empty":
        sh = {key: value[:0] for key, value in sh.items()}
    if case == "dead":
        sh["flag"] |= np.uint32(1 << FLAG["dead"])
    elif case == "unselected":
        sh["flag"][:] = 0
    out = combo.panel_hits(sh)
    assert out.keys() == reference.keys()
    for key, value in out.items():
        assert value.shape == (0,), key
        assert value.dtype == reference[key].dtype, key


@pytest.mark.parametrize("case", ["empty", "dead", "noisy", "one_hit"])
def test_time_clusters_empty_result_preserves_schema(case):
    reference = linefind.time_clusters(_sh([0, 0], [1, 2], [10, 11]))
    sh = _sh([0], [1], [10])
    if case == "empty":
        sh = {key: value[:0] for key, value in sh.items()}
    if case in ("dead", "noisy"):
        sh["flag"] |= np.uint32(1 << FLAG[case])
    out = linefind.time_clusters(sh)
    assert out.keys() == reference.keys()
    for key, value in out.items():
        assert value.shape == (0,), key
        assert value.dtype == reference[key].dtype, key


def test_dumper_trees_are_found_under_trkdump_or_one_module_directory(tmp_path):
    """hitreco.tree: the T-VR tables' trkdump/ directory, or the single module directory of another dump
    (dump_recostages_ce.fcl writes stagedump/events); none or several is an error."""
    import uproot
    from trkfastsim.reco import hitreco
    cols = {"run": np.array([1430]), "subrun": np.array([7]), "event": np.array([3])}
    for name, dirs in (("vr.root", ["trkdump"]), ("stages.root", ["stagedump"]), ("two.root", ["a", "b"]),
                       ("none.root", [])):
        with uproot.recreate(tmp_path / name) as f:
            for d in dirs:
                f[f"{d}/events"] = cols
                f[f"{d}/conditions"] = {"tdcLSB": np.array([0.03])}
            if not dirs:
                f["other"] = cols
    assert int(hitreco.tree(tmp_path / "vr.root").arrays()["event"][0]) == 3
    assert int(hitreco.tree(tmp_path / "stages.root").arrays()["subrun"][0]) == 7
    assert float(hitreco.tree(tmp_path / "stages.root", "conditions").arrays()["tdcLSB"][0]) == 0.03
    for bad in ("two.root", "none.root"):
        with pytest.raises(KeyError):
            hitreco.tree(tmp_path / bad)
