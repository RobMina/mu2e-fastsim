"""Incident-frame diagnostics use aligned identities and folded simulation timing."""
import json

import numpy as np
import pytest

from calofastsim.showers.validity_report import diagnostics, observables, replay_comparison
from test_showers_response import source


def sample_saved():
    library, _, _ = source()
    library["entry_event_row"] = np.array([0])
    library["entry_time"] = np.array([2695.])
    timing = dict(condition_pbtime=np.full(1000, -200.), condition_mbtime=np.full(1000, 1695.),
                  condition_event_length=np.full(1000, 1700.), condition_spill=np.ones(1000, int),
                  condition_marker=np.full(1000, 200.))
    saved = dict(query_event_rows=np.array([0]), event_row=np.arange(1000))
    for name in ("stock1", "stock2", "stock", "baseline", "crystal"):
        for key, value in (("energy", 80.), ("time", 1003.), ("x", 5.), ("y", 2.), ("z", 10.)):
            a = np.full(1000, 0. if key == "energy" else np.nan)
            a[0] = value
            saved[name + "_" + key] = a
    return library, saved, timing


def test_folded_entry_time_not_shifted_by_mc_pbtime_twice():
    library, saved, timing = sample_saved()
    result = diagnostics(library, saved, timing)
    h = result["arms"]["stock"]["shapes"]["Cluster time minus folded entry [ns]"]
    assert h["paired_mean"]["fast"] == h["paired_mean"]["stock"] == 3.
    assert h["outside_range"] == dict(fast=0, stock=0)
    assert result["arms"]["stock"]["common_distance_mm"]["difference"] == 0
    json.dumps(result, allow_nan=False)


def test_common_selection_and_unavailable_strata():
    library, saved, timing = sample_saved()
    saved["crystal_energy"][0] = 0
    result = diagnostics(library, saved, timing)
    for arm in result["arms"].values():
        assert arm["common_selected"] == 0
        assert arm["common_distance_mm"]["difference"] is None
        assert arm["strata"]["radius_450_550"]["acceptance"]["ratio"] is None
    json.dumps(result, allow_nan=False)


def test_radial_and_tangential_coordinates_rotate_together():
    entry = dict(position=np.array([[0., 5., 100.]]), energy=np.array([100.]), time=np.array([1000.]))
    cl = dict(energy=np.array([80.]), time=np.array([1003.]), x=np.array([-2.]), y=np.array([7.]))
    out = observables(cl, entry, np.array([1000.]), np.zeros(2))
    np.testing.assert_allclose(out["radial"], [2.])
    np.testing.assert_allclose(out["tangent"], [2.])
    assert out["truth_ep"][0] == pytest.approx(80 / np.sqrt(100 ** 2 - .51099895 ** 2))


def test_query_and_geometry_mismatches_rejected():
    library, saved, timing = sample_saved()
    saved["query_event_rows"] = np.array([1])
    with pytest.raises(ValueError, match="identities"):
        diagnostics(library, saved, timing)
    saved["query_event_rows"] = np.array([0])
    library["geometry_disk_origin_x"][1] += 1
    with pytest.raises(ValueError, match="common disk"):
        diagnostics(library, saved, timing)


def test_cpu_gpu_difference_report_does_not_claim_exactness(tmp_path):
    _, saved, _ = sample_saved()
    reports = {"arms": {name: {"timing": {"warm_reco_seconds": 1.}} for name in ("stock", "baseline", "crystal")}}
    paths = [tmp_path / "cpu", tmp_path / "gpu"]
    for path in paths:
        path.mkdir()
        (path / "numbers.json").write_text(json.dumps(reports))
        np.savez(path / "clusters.npz", **saved)
    saved["baseline_energy"][0] += 1e-5
    np.savez(paths[1] / "clusters.npz", **saved)
    result = replay_comparison(*paths)
    assert result["stock"]["exact"]["energy"]
    assert not result["baseline"]["exact"]["energy"]
    assert result["baseline"]["selection_equal"]
    assert result["baseline"]["differences"]["energy"]["changed_finite_values"] == 1
    assert result["baseline"]["differences"]["energy"]["max_abs"] == pytest.approx(1e-5)
