"""Fixed-population C4 comparisons, including paired errors and empty response arms."""
import awkward as ak
import numpy as np
import pytest

from crvfastsim.veto import VetoCard
from crvfastsim.veto.paired import align, compare, main, paired_rate


def sample(times=None):
    times = [[900.], [], [900.], [900.]] if times is None else times
    t = ak.Array(times)
    co = ak.zip(dict(time=t, timeStart=t - 1, timeEnd=t + 1, PEs=ak.ones_like(t) * 50,
                     nLayers=ak.ones_like(t) * 3, sectorType=ak.ones_like(t, dtype=np.int64)))
    return ak.zip(dict(run=[1] * 4, subrun=[1, 1, 2, 2], event=[1, 2, 1, 2],
                       p_mid=ak.Array([[100.]] * 4), t_mid=ak.Array([[1000.]] * 4), coincs=co), depth_limit=1)


def test_identical_reordered_has_zero_paired_error():
    a = sample()
    r = compare(a, VetoCard(), a[[3, 0, 2, 1]], sectors=[1, 2, 3])
    assert r["vetoed"]["k"] == 3
    assert r["inefficiency"]["k"] == 1
    assert r["paired_inefficiency"]["ratio"] == 1
    assert r["paired_inefficiency"]["ratio_se"] == 0
    assert r["paired_inefficiency"]["difference_se"] == 0
    assert r["by_sector"]["2"]["stock"] == 0
    assert r["by_sector"]["2"]["ratio"] is None


def test_empty_fast_arm_keeps_denominator():
    a = sample()
    fast = ak.with_field(a, a.coincs[:, :0], "coincs")
    r = compare(a, VetoCard(), fast)
    assert r["fast"]["inefficiency"]["k"] == 4
    assert r["paired_inefficiency"]["n"] == 4
    assert r["paired_inefficiency"]["ratio"] == 4
    assert r["discordant"] == {"stock_only": 3, "fast_only": 0}


def test_gate_migration_does_not_change_population():
    a = sample()
    fast = sample([[300.], [], [900.], [900.]])
    r = compare(a, VetoCard(gate_start_lo=420), fast)
    assert r["selected"] == 4
    assert r["fast"]["gate_failed_tracks"] == 1
    assert r["fast"]["inefficiency"]["n"] == 4


def test_duplicate_or_missing_events_rejected():
    a = sample()
    with pytest.raises(ValueError, match="every cached"):
        align(a, a[:3])
    with pytest.raises(ValueError):
        align(a, a[[0, 0, 2, 3]])


def test_ratio_error_uses_paired_shards():
    result = paired_rate([1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 1])
    assert result["ratio"] == 1.5
    assert result["ratio_se"] == .5
    assert result["difference"] == .25
    assert result["difference_se"] == .25
    one = paired_rate([1], [0], [0])
    assert one["ratio"] == 0 and one["ratio_se"] is None
    empty = paired_rate([], [], [])
    assert empty["n"] == 0 and empty["ratio"] is None


def test_multiple_tracks_exclusive_sector_attribution():
    a = sample()
    a = ak.with_field(a, ak.Array([[1000., 1100.], [], [1000.], [1000.]]), "t_mid")
    a = ak.with_field(a, ak.Array([[100., 101.], [], [102.], [103.]]), "p_mid")
    r = compare(a, VetoCard(), a)
    assert r["selected"] == 4 and r["vetoed"]["k"] == 3
    assert r["by_sector"]["1"]["stock"] == 3


def test_cli_stock_only(tmp_path):
    import json
    a = sample()
    ak.to_parquet(a, tmp_path / "sel.parquet")
    (tmp_path / "card.json").write_text(json.dumps({"card": {"sha": "informational"}}))
    main([str(tmp_path / "sel.parquet"), "--card", str(tmp_path / "card.json"), "--outdir", str(tmp_path / "out")])
    r = json.loads((tmp_path / "out/numbers.json").read_text())
    assert r["vetoed"]["k"] == 3 and r["inefficiency"]["k"] == 1
    assert "fast" not in r
    assert (tmp_path / "out/crv_veto.png").exists()
