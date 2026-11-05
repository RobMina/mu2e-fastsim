"""Source-only comparisons keep flows and normalise independent draws explicitly."""
import numpy as np
import pytest

from beamfastsim import rpccompare as C
from beamfastsim import rpcspec as S


def test_identical_shapes_and_flows():
    r = C.shape_comparison([5, 100, 300, 20], [10, 200, 600, 40])
    assert r["chi2"] == 0 and r["scale"] == .5
    assert r["stock_entries"] == 425 and r["fast_entries"] == 850
    assert r["flow_stock"] == [5, 20]
    assert r["pulls"] == [0, 0, 0, 0]


def test_flow_difference_is_not_dropped():
    r = C.shape_comparison([0, 1000, 1000, 0], [500, 1000, 1000, 500])
    assert r["chi2"] > 0 and r["pulls"][0] > 0


@pytest.mark.parametrize("stock,fast", [([0, 0], [1, 2]), ([1], [1, 2]), ([-1, 2], [1, 2]),
                                         ([np.nan, 2], [1, 2])])
def test_invalid_shapes(stock, fast):
    with pytest.raises(ValueError):
        C.shape_comparison(stock, fast)


def test_source_figure_and_numbers(tmp_path):
    import json
    import uproot
    rng = np.random.default_rng(44)
    source = S.internal_pair(rng, S.PhotonSpectrum().sample(rng, 1000))
    np.savez(tmp_path / "source.npz", **source)
    with uproot.recreate(tmp_path / "stock.root") as root:
        for key, (histname, _) in C.HISTOGRAMS.items():
            root["generate/RPCGun/" + histname] = np.histogram(source[key], bins=np.linspace(0, 140, 29))
    report = C.compare(tmp_path / "stock.root", tmp_path / "source.npz", tmp_path / "out")
    assert report["events"] == 1000
    assert all(r["chi2"] == 0 for r in report["shapes"].values())
    assert (tmp_path / "out/rpc_source.png").is_file()
    assert json.loads((tmp_path / "out/numbers.json").read_text())["events"] == 1000
