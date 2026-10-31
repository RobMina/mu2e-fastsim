"""Paired-frame errors, observed-channel aggregation and source-subset separation."""
import numpy as np
import pytest

from crvfastsim.veto.mixframes import MIXERS
from crvfastsim.veto.replay_metrics import paired_stats, pulse_compare, strongest
from crvfastsim.veto.replay_subset import select


def test_paired_covariance_and_empty_frames():
    f, s = np.array([0, 1, 0.5, 0]), np.array([0, 1, 0, 0])
    r = paired_stats(f, s)
    assert r["delta"] == 0.125
    assert r["delta_err_paired"] == pytest.approx(np.std(f - s, ddof=1) / 2)
    assert paired_stats(f, f)["delta_err_paired"] == 0
    assert paired_stats([0], [0])["delta_err_paired"] is None
    with pytest.raises(ValueError):
        paired_stats([], [])


def test_strongest_channel_diagnostics():
    p = {"ev": np.array([1, 1, 1, 2]), "chan": np.array([0, 0, 4400, 0]),
         "pe": np.array([3., 6., 12., 9.]), "time": np.array([10., 11., 15., 20.])}
    best = strongest(p)
    np.testing.assert_array_equal(best["pe"], [6, 12, 9])
    r = pulse_compare(p, p)
    assert r["common_channels"] == 3 and r["fast_only_channels"] == 0
    assert r["sectors"]["T1"]["matched_dt_q10_50_90"] == [0, 0, 0]
    assert r["sectors"]["T2"]["matched_pe_ratio_q10_50_90"] == [1, 1, 1]


def test_subsets_disjoint_and_reproducible():
    names = [f"dts.mu2e.{sp}.test.{n:05d}" for sp in MIXERS for n in range(20)]
    a, b = select(names, 4, 0, 123), select(names, 4, 1, 123)
    assert a == select(list(reversed(names)), 4, 0, 123)
    for sp in MIXERS:
        assert not set(a[sp]) & set(b[sp])
    with pytest.raises(ValueError):
        select(names, 4, 5, 123)


def test_optional_pulse_sink_preserves_clusters(monkeypatch):
    from crvfastsim.veto import fastframes as ff
    from crvfastsim.reco import pulsefit
    from types import SimpleNamespace
    pulse = {"time": np.array([1.]), "wf": np.array([0]), "pe": np.array([20.]), "seq": np.array([0])}
    monkeypatch.setattr(pulsefit, "reconstruct", lambda *args: {k: v.copy() for k, v in pulse.items()})
    monkeypatch.setattr(ff.C, "find_event", lambda *args: [])
    w = {k: np.array([0]) for k in ("n", "start", "ped", "area", "height", "chan", "sipm", "bar")}
    w.update(adc=np.zeros((1, 12)), ev=np.array([17]))
    cond = SimpleNamespace(timeoffset=np.array([2.]), status=np.array([0]))
    a = ff._clusters(w, cond, None, None, {})
    sink = []
    b = ff._clusters(w, cond, None, None, {}, sink)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert sink[0]["time"][0] == 3
    assert sink[0]["ev"][0] == 17


def test_shared_table_yield_diagnostic_is_copy_not_default_change():
    from crvfastsim.veto.fastframes import run_config
    from crvfastsim.veto.replay_fast import stock_shared_table_yields
    config = run_config("run1a_v01")[0]
    diagnostic = stock_shared_table_yields(config)
    assert diagnostic.sectors["T1"].yield_per_mev == diagnostic.sectors["T2"].yield_per_mev == 39794
    assert config.sectors["T2"].yield_per_mev == 30482
    assert run_config("run1a_v01")[0].sectors["T2"].yield_per_mev == 30482


def test_excluded_prior_subset_manifest(tmp_path):
    import json
    from crvfastsim.veto.replay_subset import main
    names = [f"dts.mu2e.{sp}.test.{n:05d}" for sp in MIXERS for n in range(10)]
    inventory = tmp_path / "inventory.txt"
    inventory.write_text("\n".join(names))
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    base = [str(inventory), "--pnfs", "/dcache-test/example", "--per-species", "2", "--block", "0"]
    main(base + ["--out", str(first)])
    main(base + ["--out", str(second), "--exclude", str(first.with_suffix(".json"))])
    a = json.loads(first.with_suffix(".json").read_text())["selected"]
    b = json.loads(second.with_suffix(".json").read_text())["selected"]
    for sp in MIXERS:
        assert not set(a[sp]) & set(b[sp])
