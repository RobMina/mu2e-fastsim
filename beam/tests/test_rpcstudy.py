"""RPC study tests on synthetic windows and transport outputs."""
import numpy as np
import pytest

from beamfastsim import rpcstudy as R


def test_window_occurrences_are_not_collapsed(tmp_path):
    n = 6
    p = tmp_path / "window.npz"
    np.savez(p, xstop=np.arange(n), ystop=np.zeros(n), zstop=np.full(n, 6000.), tstop=np.full(n, 400.),
             pdg=np.full(n, -211), at_rest=np.ones(n, bool), foil=np.zeros(n, int), codestop=np.full(n, 78),
             production_ifile=np.repeat([0, 1], 3), rpc_event_occurrence=np.tile(np.arange(3), 2))
    result = R.starts(p, tmp_path / "starts.npz", repeats=3, seed=3)
    assert result["captures"] == 18
    with np.load(tmp_path / "starts.npz") as a:
        assert np.array_equal(a["x"], np.tile(np.arange(n), 3))
        assert a["positron_p"].size == 18
        assert result["per_shard"] == {"0": 9, "1": 9}
        assert np.all(a["time"] == 400.)


def test_ratio_is_paired_per_shard_and_counts_empty_arm():
    r = R.clustered_ratio([10, 20], [12, 22], [100, 100])
    assert r["ratio"] == pytest.approx(34 / 30)
    assert r["difference"] == pytest.approx(.02)
    assert r["difference_error"] == 0
    assert r["ratio_error"] > 0
    r = R.clustered_ratio([10, 20], [0, 0], [100, 100])
    assert r["exposure"] == 200 and r["fast_yield"] == 0 and r["ratio"] == 0
    assert R.clustered_ratio([0, 0], [0, 0], [100, 100])["ratio"] is None
    assert R.clustered_ratio([10], [12], [100])["ratio_error"] is None


def test_histograms_include_flows_and_empty_shards():
    h = R.hist_by_shard(np.array([-2., .5, 4.]), np.ones(3, bool), np.array([0, 0, 2]),
                       np.array([0, 1, 2]), np.array([0., 1., 2.]))
    assert np.array_equal(h, [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])


def test_identical_shapes_have_zero_pulls_or_undefined_errors():
    s = np.array([[0., 30, 40, 0], [1., 45, 15, 1]])
    r = R.shape_report(s, s * 2, np.array([0., 1., 2.]))
    assert r["chi2"] == 0 and r["scale"] == .5
    assert all(v == 1 for v in r["ratio"])
    assert r["zero_variance_nonzero_difference"] == 0


def test_identical_paired_histograms_have_zero_variance():
    s = np.array([[0., 30, 40, 0], [1., 45, 15, 1]])
    joint = np.diag(s.sum(axis=0))
    r = R.shape_report(s, s, np.array([0., 1., 2.]), joint=joint)
    assert r["chi2"] == 0 and r["ndf"] == 0
    assert np.allclose(r["difference_error"], 0, atol=1e-6)


def test_shape_variance_contains_source_draw_noise_with_identical_shards():
    s = np.array([[0., 20, 40, 0], [0., 20, 40, 0]])
    f = np.array([[0., 30, 30, 0], [0., 30, 30, 0]])
    r = R.shape_report(s, f, np.array([0., 1., 2.]))
    assert r["shard_difference_error"][1] == 0
    assert r["difference_error"][1] > 0
    assert r["chi2"] > 0


def fixtures(tmp_path, source=False):
    n = 8
    ids = np.repeat(np.arange(4), 2)
    p = np.full(n, 100.)
    x = np.column_stack([np.full(n, -3904.), np.full(n, 100.), np.full(n, 8539.9)])
    u = np.column_stack([np.zeros(n), np.zeros(n), np.ones(n)])
    raw = dict(x=x[:, 0], y=x[:, 1], z=x[:, 2], px=np.zeros(n), py=np.zeros(n), pz=p, time=np.full(n, 500.))
    ref = tmp_path / "ref.npz"
    np.savez(ref, **raw, gencount=np.full(4, 10), ifile=ids, isvd_front=np.ones(n, bool),
             xvd_front=x[:, 0], yvd_front=x[:, 1], zvd_front=x[:, 2], pxvd_front=np.zeros(n),
             pyvd_front=np.zeros(n), pzvd_front=p)
    fast = tmp_path / "fast.npz"
    data = dict(idx=np.arange(n), x0=x, u0=u, p0=p, t0=raw["time"],
                fate=np.array([1, 1, 1, 0, 1, 1, 1, 0]), x=x, u=u,
                T=np.hypot(p, .51099895) - .51099895, wall=np.array(1.))
    np.savez(fast, **data)
    src = tmp_path / "starts.npz"
    if source:
        np.savez(src, **raw, shard=ids)
    return ref, fast, src, data


def test_saved_start_comparison_is_conditional_and_all_rows_retained(tmp_path):
    ref, fast, _, data = fixtures(tmp_path)
    report = R.compare(ref, fast, [0, 1])
    assert "conditional" in report["scope"]
    y = report["selections"]["vd13"]["yield_"]
    assert y["exposure"] == 4 and y["stock_count"] == 4 and y["fast_count"] == 3
    assert y["ratio"] == .75 and y["ratio_error"] == .25
    data["x0"] = data["x0"] + 1
    np.savez(fast, **data)
    with pytest.raises(ValueError, match="share stock starts"):
        R.compare(ref, fast, [0, 1])


def test_generated_comparison_uses_catalog_denominator_not_kept_rows(tmp_path):
    ref, fast, source, _ = fixtures(tmp_path, source=True)
    report = R.compare(ref, fast, [0, 1], starts_path=source)
    y = report["selections"]["vd13"]["yield_"]
    assert y["exposure"] == 20
    assert y["stock_yield"] == .2 and y["fast_yield"] == .75
    assert "not inclusive" in report["scope"]


def test_missing_fast_rows_are_not_silently_removed(tmp_path):
    ref, fast, _, data = fixtures(tmp_path)
    data["idx"] = np.zeros(8, int)
    np.savez(fast, **data)
    with pytest.raises(ValueError, match="cover each"):
        R.compare(ref, fast, [0, 1])


def test_tracker_front_disk_radius_applies_to_fast_crossings(tmp_path):
    ref, fast, _, data = fixtures(tmp_path)
    data["x"] = data["x"].copy()
    data["x"][0, 1] = 900.
    np.savez(fast, **data)
    report = R.compare(ref, fast, [0, 1])
    assert report["selections"]["vd13"]["yield_"]["fast_count"] == 2
    assert report["vd_radius_mm"] == 850.1


def test_helix_cut_is_optional_and_keeps_denominators(tmp_path, monkeypatch):
    ref, fast, _, _ = fixtures(tmp_path)
    def radius(q, geom, field):
        q["rmax"] = np.tile([400., 500.], 4)
    monkeypatch.setattr(R, "add_helix_radius", radius)
    report = R.compare(ref, fast, [0, 1], geom=object(), field=object())
    y = report["selections"]["fiducial"]["yield_"]
    assert y["exposure"] == 4 and y["stock_count"] == 2 and y["fast_count"] == 1
    assert report["selections"]["fiducial"]["rmax_min"] == 450.


def test_plot(tmp_path):
    ref, fast, _, _ = fixtures(tmp_path)
    report = R.compare(ref, fast, [0, 1])
    R.plot(report, tmp_path / "plot.png")
    assert (tmp_path / "plot.png").is_file()
