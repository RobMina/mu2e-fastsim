"""Synthetic paired-shard regressions; no transport or external data required."""
import json

import numpy as np
import pytest

from beamfastsim import tscompare as ts


def tables(shards=(1, 1, 2, 2), pdg=None, stock=None, fast=None):
    shard = np.asarray(shards, dtype=np.int64)
    n = len(shard)
    pdg = np.full(n, 13, dtype=np.int64) if pdg is None else np.asarray(pdg, dtype=np.int64)
    reached = np.ones(n, bool) if stock is None else np.asarray(stock, bool)
    fate = np.ones(n, dtype=np.int64) if fast is None else np.asarray(fast, dtype=np.int64)
    output_pdg = np.where(np.abs(pdg) == 211, -13 * np.sign(pdg), pdg)
    p = np.tile([0., 0., 65.], (n, 1))
    x = np.zeros((n, 3))
    ids = np.column_stack((np.ones(n, dtype=np.int64), shard, np.arange(n), np.full(n, 7)))
    r = dict(id=ids, shard=shard, x=x, p=p, t=np.full(n, 500.), pdg=pdg,
             stock_reached=reached, stock_x=x.copy(), stock_p=p.copy(),
             stock_t=np.full(n, 600.), stock_pdg=output_pdg)
    f = dict(id=ids.copy(), shard=shard.copy(), x=x.copy(), p=p.copy(),
             t=np.full(n, 600.), pdg=output_pdg.copy(), fate=fate, seconds=np.array(.2))
    return r, f


def subset(a, indices):
    return {k: v[indices] if v.ndim else v for k, v in a.items()}


def mu(result, charge=-1, name="mu-"):
    return result["charges"][str(charge)]["muons"][name]


def test_paired_identity_and_empty_charge():
    r, f = tables(stock=[1, 0, 1, 0], fast=[1, 0, 1, 0])
    result = ts.compare(r, f)
    m = mu(result)
    assert (m["stock"], m["fast"], m["difference"], m["ratio"]) == (2, 2, 0., 1.)
    assert m["difference_se"] == m["ratio_se"] == m["pull"] == 0.
    for h in m["histograms"].values():
        assert h["chi2_diagonal"] == 0.
        assert np.all(np.asarray(h["difference_se"]) == 0.)
        assert np.all(np.asarray(h["pull"]) == 0.)
    empty = mu(result, 1, "mu+")
    assert empty["stock"] == empty["fast"] == 0
    assert empty["difference"] is None and empty["ratio"] is None
    json.dumps(result, allow_nan=False)


def test_correlated_migration_uses_shards_not_independent_particles():
    # All particles in one shard migrate together: per-event error is too small.
    r, f = tables(shards=np.repeat([1, 2, 3], 4))
    f["p"][:4, 2] = 75.
    m = mu(ts.compare(r, f))
    assert m["difference_se"] == 0.  # equal inclusive yield
    h = m["histograms"]["momentum"]
    assert h["difference"][6] == pytest.approx(-1 / 3)
    assert h["difference_se"][6] == pytest.approx(1 / 3)
    assert h["difference_se"][7] == pytest.approx(1 / 3)
    assert h["pull"][6:8] == pytest.approx([-1., 1.])
    assert h["chi2_diagonal"] == pytest.approx(2.)
    assert h["stock_shard"] == [[0] * 6 + [4] + [0] * 13] * 3


def test_unequal_shard_estimating_equations():
    r, f = tables(shards=[1, 2, 2, 2, 3, 3], stock=[1, 1, 0, 0, 1, 0], fast=[0, 1, 1, 1, 1, 1])
    m = mu(ts.compare(r, f))
    n, s, a = np.array([1, 3, 2]), np.array([1, 1, 1]), np.array([0, 3, 2])
    delta, ratio = 2 / 6, 5 / 3
    de = np.sqrt(1.5 * np.sum((a - s - delta * n) ** 2)) / n.sum()
    re = np.sqrt(1.5 * np.sum((a - ratio * s) ** 2)) / s.sum()
    assert m["difference"] == pytest.approx(delta)
    assert m["ratio"] == pytest.approx(ratio)
    assert m["difference_se"] == pytest.approx(de)
    assert m["ratio_se"] == pytest.approx(re)
    assert m["stock_fraction"] == .5 and m["fast_fraction"] == pytest.approx(5 / 6)


def test_exact_join_reorders_and_uses_reference_start_time():
    r, f = tables(pdg=[-211, -13, 211, 13])
    r["t"] = np.array([500., 700., 900., 1100.])
    r["stock_t"] = f["t"] = r["t"] + 65.
    f["x"] += 1000  # positions are never a matching key
    permuted = subset(f, [3, 0, 2, 1])
    assert ts.compare(r, f) == ts.compare(r, permuted)
    for charge, name in [(-1, "mu-"), (1, "mu+")]:
        h = mu(ts.compare(r, permuted), charge, name)["histograms"]["elapsed_time"]
        assert h["stock"][3] == h["fast"][3] == 2


@pytest.mark.parametrize("column", [0, 1, 2, 3])
def test_no_fuzzy_id_match(column):
    r, f = tables()
    f["id"][0, column] += 1000
    with pytest.raises(ValueError, match="IDs absent"):
        ts.compare(r, f)


@pytest.mark.parametrize("arm", [0, 1])
def test_duplicate_id_rejected(arm):
    pair = tables()
    pair[arm]["id"][1] = pair[arm]["id"][0]
    with pytest.raises(ValueError, match="duplicate IDs"):
        ts.compare(*pair)


def test_missing_rows_rejected_even_within_explicit_selection():
    r, f = tables()
    partial = subset(f, [0, 1])
    with pytest.raises(ValueError, match="missing fast rows"):
        ts.compare(r, partial)
    assert ts.compare(r, partial, [1])["ninput"] == 2
    assert ts.compare(r, f, [1])["ninput"] == 2
    with pytest.raises(ValueError, match="missing fast rows"):
        ts.compare(r, subset(f, [0]), [1])
    with pytest.raises(ValueError, match="selected shards"):
        ts.compare(r, f, [9])
    f["shard"][0] = 2
    with pytest.raises(ValueError, match="shard mismatch"):
        ts.compare(r, f)


def test_empty_success_shards_parent_inventory_and_empty_arms_kept():
    r, f = tables(stock=[1, 0, 0, 0], fast=[0, 0, 0, 0])
    r.update(parent_shard=np.array([1, 2, 3]), parent_shard_count=np.array([2, 2, 20]))
    result = ts.compare(r, f)
    assert result["shards"] == [1, 2, 3]
    assert result["charges"]["-1"]["ninput_shard"] == [2, 2, 0]
    m = mu(result)
    assert m["stock_shard"] == [1, 0, 0] and m["fast_shard"] == [0, 0, 0]
    assert m["stock_fraction"] == .25  # not parent event counts
    assert m["difference_se"] == pytest.approx(np.sqrt(.75) / 4)  # K=3, including empty shard
    assert m["ratio"] == 0.
    # Select a known parent shard with no input rows, keeping that shard in K.
    empty = ts.compare(r, subset(f, []), [3])
    assert empty["ninput"] == 0 and empty["nshards"] == 1
    assert mu(empty)["difference_se"] is None


def test_entirely_empty_tables():
    r, f = tables(shards=[])
    result = ts.compare(r, f)
    assert result["nshards"] == result["ninput"] == 0
    assert mu(result)["ratio"] is None
    json.dumps(result, allow_nan=False)


def test_charge_mapping_groups_on_input_and_output_muons_separately():
    r, f = tables(pdg=[211, -211, 13, -13], stock=[1, 1, 1, 0])
    result = ts.compare(r, f)
    assert mu(result, 1, "mu+")["stock"] == 1
    assert mu(result, 1, "mu+")["fast"] == 2
    assert mu(result, -1, "mu-")["stock"] == 2
    assert result["charges"]["1"]["ninput"] == 2
    assert any("Scarce positive" in w for w in result["warnings"])
    # An unexpected opposite-charge endpoint stays in its INPUT cohort.
    f["pdg"][0] = 13
    result = ts.compare(r, f)
    assert mu(result, 1, "mu-")["fast"] == 1
    assert mu(result, -1, "mu-")["fast"] == 2
    f["pdg"][0] = 211  # DS pions are not counted as muons
    assert ts.compare(r, f)["charges"]["1"]["ds"]["fast"] == 2
    assert mu(ts.compare(r, f), 1, "mu+")["fast"] == 1


def test_nonzero_difference_zero_variance_never_fabricates_significance():
    r, f = tables(stock=[1, 0, 1, 0])
    m = mu(ts.compare(r, f))
    assert m["difference"] == .5 and m["difference_se"] == 0.
    assert m["ratio"] == 2. and m["ratio_se"] == 0.
    assert m["pull"] is None and m["zero_variance_nonzero_difference"]
    h = m["histograms"]["momentum"]
    assert h["pull"][6] is None and h["chi2_diagonal"] is None
    assert h["zero_variance_nonzero_difference"][6]
    assert mu(ts.compare(r, f, [1]))["difference_se"] is None


def test_histogram_flows_last_edge_and_finite_states():
    r, f = tables()
    r["stock_p"][:, 2] = f["p"][:, 2] = [0., 200., 201., 65.]
    r["stock_t"] = f["t"] = r["t"] + [-1., 500., 501., 65.]
    m = mu(ts.compare(r, f))
    assert m["histograms"]["momentum"]["stock_flow"] == [0, 1]
    assert m["histograms"]["momentum"]["stock"][-1] == 1
    assert m["histograms"]["elapsed_time"]["stock_flow"] == [1, 1]
    assert m["histograms"]["elapsed_time"]["stock"][-1] == 1
    f["p"][0, 2] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        ts.compare(r, f)
    f["fate"][0] = 0  # failed states need not be finite
    ts.compare(r, f)


def test_cli_and_plot(tmp_path):
    r, f = tables(stock=[1, 0, 1, 0])
    rp, fp = tmp_path / "reference.npz", tmp_path / "fast.npz"
    np.savez(rp, **r); np.savez(fp, **f)
    output = tmp_path / "comparison"
    ts.main(["--reference", str(rp), "--fast", str(fp), "--out", str(output), "--label", "held-out"])
    result = json.loads((output / "numbers.json").read_text())
    assert result["label"] == "held-out" and result["ninput"] == 4
    assert (output / "ts.png").read_bytes().startswith(b"\x89PNG")
    ts.plot(ts.compare(r, f), tmp_path / "development.png")
    with pytest.raises(ValueError, match="label"):
        ts.plot(result, tmp_path / "bad.png", "precision")


def test_parent_run_shard_inventory_keeps_empty_units_and_distinct_runs():
    r, f = tables(shards=[1, 1, 1, 1])
    r["id"][2:, 0] = f["id"][2:, 0] = 2
    r.update(parent_shard_ids=np.array([[1, 1], [2, 1], [3, 1], [3, 2]]),
             parent_shard_event_counts=np.array([10, 20, 30, 40]))
    result = ts.compare(r, f, [1])
    assert result["independent_shards"] == [[1, 1], [2, 1], [3, 1]]
    assert result["nshards"] == 3
    assert result["charges"]["-1"]["ninput_shard"] == [2, 2, 0]
    assert mu(result)["stock_shard"] == [2, 2, 0]
    assert ts.compare(r, f)["nshards"] == 4


def test_proportional_counts_zero_variance_not_roundoff_significance():
    r, f = tables(shards=np.repeat([1, 2], 49), stock=np.zeros(98, bool), fast=np.zeros(98, int))
    f["fate"][[0, 49]] = 1
    m = mu(ts.compare(r, f))
    assert m["difference"] == pytest.approx(1 / 49)
    assert m["difference_se"] == 0. and m["pull"] is None
    assert m["zero_variance_nonzero_difference"]
    assert m["ratio"] is None and m["ratio_se"] is None


@pytest.mark.parametrize("key,value", [
    ("id", np.ones((4, 4), dtype=float)),
    ("p", np.zeros((4, 2))),
    ("stock_reached", np.full(4, 2)),
    ("pdg", np.full(4, 11, dtype=int)),
])
def test_malformed_reference_rejected(key, value):
    r, f = tables()
    r[key] = value
    with pytest.raises(ValueError):
        ts.compare(r, f)


def test_only_empty_parent_shards_remain_json_safe():
    r, f = tables(shards=[])
    r.update(parent_shard_ids=np.array([[4, 1], [4, 2]]), parent_shard_event_counts=np.array([10, 10]))
    result = ts.compare(r, f)
    assert result["independent_shards"] == [[4, 1], [4, 2]]
    assert mu(result)["stock_shard"] == [0, 0]
    assert mu(result)["difference_se"] is None
    json.dumps(result, allow_nan=False)


def test_large_ids_remain_exact_beyond_float_precision():
    r, f = tables()
    r["id"][:, 3] = np.arange(4, dtype=np.int64) + 2 ** 54
    f["id"] = r["id"].copy()
    assert mu(ts.compare(r, subset(f, [3, 2, 1, 0])))["difference"] == 0.
    f["id"][0, 3] += 10
    with pytest.raises(ValueError, match="IDs absent"):
        ts.compare(r, f)


def test_run1a_label_is_explicit_without_changing_statistics():
    r, f = tables()
    a, b = ts.compare(r, f, geometry_label="Run1A"), ts.compare(r, f)
    assert a["geometry_label"] == "Run1A"
    assert b["geometry_label"] == "Run1B"
    assert a["charges"] == b["charges"]
    assert "not Run1A validation" not in a["scope"]
    with pytest.raises(ValueError, match="geometry_label"):
        ts.compare(r, f, geometry_label="other")


def test_projected_cohort_metadata_and_fates_survive_comparison():
    r, f = tables()
    r.update(nested_id=np.array([[1, 1, 3, 8]], dtype='i8'),
             source_parent_ancestry_incomplete=np.array([True, False, False, False]),
             parent_shard_ids=np.array([[1, 1], [1, 2]], dtype='i8'),
             parent_shard_event_counts=np.array([5, 5], dtype='i8'))
    result = ts.compare(r, f, geometry_label='Run1A')
    assert result['reference_metadata'] == {'cohort_definition': 'known root entrance lineages',
                                            'nested_tracks_represented_by_ancestor': 1,
                                            'upstream_ancestry_incomplete_sources': 1,
                                            'parent_events': 10}
    assert result['fast_fate_counts'] == {'1': 4}


def test_sparse_fast_positive_arm_warns_even_with_fifteen_stock():
    r, f = tables(shards=np.arange(15), pdg=np.full(15, -13), fast=np.r_[np.ones(8), np.zeros(7)])
    result = ts.compare(r, f)
    assert result['charges']['1']['headline']['stock'] == 15
    assert result['charges']['1']['headline']['fast'] == 8
    assert any('Scarce positive' in w for w in result['warnings'])
