"""Full-event accounting and measurement-driver regressions."""
import numpy as np

from calofastsim.showers.measure_chain import compare
from calofastsim.showers.reconstruct import leading, report, steps_for_reco
from calofastsim.showers.prepare_chain import stock_clusters
from calofastsim.showers.export import EVENT, keys


def test_generated_denominator_and_separate_ratio_error():
    n = 200
    arrays = {k: np.zeros(n) for k in ("fast_visible", "fast_response", "stock_visible", "stock_response", "proxy_wall", "proxy_vacuum")}
    arrays["stock_visible"][[0, 1, 100, 101]] = 20
    arrays["stock_response"][[0, 100]] = 20
    arrays["fast_visible"][[0, 100, 102]] = 20
    arrays["fast_response"][[0, 102]] = 20
    arrays["proxy_wall"][[0, 1, 100]] = 1
    arrays["proxy_vacuum"][[0, 1, 100, 101]] = 1
    report = compare(arrays, np.arange(n))
    assert report["events"] == n
    assert report["counts"]["fast_response"] == 2
    pair = report["fast_response_vs_stock_response"]
    assert pair["fast"] == pair["stock"] == .01
    assert pair["ratio"] == 1
    assert pair["fast_only"] == pair["stock_only"] == 1
    assert "ratio_error" in pair and "error" in pair
    visible = report["fast_visible_vs_stock_visible"]
    assert visible["ratio"] == .75
    assert visible["fast_only"] == 1 and visible["stock_only"] == 2


def test_single_block_smoke_reports_unknown_uncertainty():
    arrays = {k: np.ones(2) * 20 for k in ("fast_visible", "fast_response", "stock_visible", "stock_response", "proxy_wall", "proxy_vacuum")}
    report = compare(arrays, np.arange(2))
    pair = report["fast_response_vs_stock_response"]
    assert pair["error"] is None and pair["ratio_error"] is None
    assert all(x is None for x in report["energy_shape"]["variance"])


def test_leading_uses_energy_and_retains_empty_events():
    cl = dict(event=np.array([2, 0, 2]), energy=np.array([20., 30., 40.]), time=np.array([1., 2., 3.]),
              x=np.array([1., 2., 3.]), y=np.zeros(3), z=np.zeros(3))
    lead = leading(cl, 4)
    np.testing.assert_array_equal(lead["energy"], [30, 0, 40, 0])
    assert lead["time"][2] == 3 and np.isnan(lead["time"][1])
    result = report(lead, lead, np.array([0, 1, 100, 101]))
    assert result["acceptance_ratio"]["ratio"] == 1
    assert result["events"] == 4 and result["fast_selected"] == 2


def test_reco_step_mapping_keeps_generated_indices():
    draw = dict(mapped=np.array([True, False, True]), crystal=np.array([1, -1, 2]), entry=np.array([0, 0, 1]),
                energy_g4=np.array([1., 2., 3.]), energy_birks=np.array([1., 2., 3.]), time=np.array([4., 5., 6.]),
                local_position=np.zeros((3, 3)))
    handoff = dict(steps=draw, entries=dict(event_row=np.array([2, 8])))
    sim = dict(lru=np.full(3, .05), pe_per_mev=np.full((3, 2), 30.))
    out = steps_for_reco(handoff, sim)
    np.testing.assert_array_equal(out["event"], [2, 8])
    np.testing.assert_array_equal(out["crystal"], [1, 2])
    np.testing.assert_array_equal(out["energy_birks"], [1, 3])


def test_stock_clusters_full_identity_highest_cycle_and_zero(tmp_path):
    import awkward as ak
    import uproot

    ids = dict(run=np.array([1, 1, 1]), subrun=np.array([0, 1, 2]), event=np.array([1, 1, 1]))
    eventinfo = ak.Array({k: v[[2, 0, 1]] for k, v in ids.items()})
    clusters = {"caloclusters.energyDep_": ak.Array([[11., 30.], [], [20.]])}
    for name in ("time_", "cog_.fCoordinates.fX", "cog_.fCoordinates.fY", "cog_.fCoordinates.fZ"):
        clusters["caloclusters." + name] = ak.Array([[2., 3.], [], [4.]])
    table = {"evtinfo": eventinfo, "caloclusters": ak.zip(clusters, depth_limit=1)}
    path = tmp_path / "stock.root"
    with uproot.recreate(path) as out:
        out["prd/ntuple"] = table
        out["prd/ntuple"] = table
    result = stock_clusters(path, keys(ids, EVENT))
    np.testing.assert_array_equal(result["energy"], [0., 20., 30.])
    np.testing.assert_array_equal(result["time"][1:], [4., 3.])
    assert np.isnan(result["time"][0])
