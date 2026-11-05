"""Unit tests: per-shard paired yield ratio and per-generated shapes of beamfastsim.diocompare."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from beamfastsim import diocompare as DC


def test_ratio_by_shard_equal_arms_and_error_scale():
    rng = np.random.default_rng(4)
    n = np.full(200, 4000)
    ks = rng.binomial(n, 0.3)
    r = DC.ratio_by_shard(ks, ks, n)
    assert r["ratio"] == 1.0 and r["ratio_err"] == 0.0
    kf = rng.binomial(n, 0.3)
    r = DC.ratio_by_shard(ks, kf, n)
    # independent binomial arms: error close to sqrt(2 (1 - y) / (y N))
    expect = np.sqrt(2 * 0.7 / (0.3 * n.sum()))
    assert abs(r["ratio_err"] / expect - 1) < 0.2
    assert abs(r["ratio"] - 1) < 4 * expect


def test_shapes_pulls_and_chi2_for_identical_parents():
    rng = np.random.default_rng(5)
    mk = lambda m: dict(p=rng.uniform(95, 105, m), cos=rng.uniform(0.4, 0.9, m), r=rng.uniform(380, 620, m),
                        t=rng.uniform(300, 1700, m))
    s = DC.shapes(mk(50_000), mk(50_000), 100_000)
    keys = ("p", "cos", "r", "t")
    for k in keys:
        assert len(s[k]["pull"]) == len(s[k]["edges"]) - 1
    pooled = sum(s[k]["chi2"] for k in keys) / sum(s[k]["ndf"] for k in keys)
    assert 0.7 < pooled < 1.3
    assert abs(sum(s["p"]["stock"]) - 0.5) < 0.01


def _inputs(tmp_path, chunks=((3, 0), (4, 2, 1)), counts=(3, 2), empty_arm=None):
    """Write small starts, stock and final-output chunks without running transport."""
    shard = np.repeat(np.array([7, 9])[:len(counts)], counts)
    n = shard.size
    starts, ref = tmp_path / "starts.npz", tmp_path / "stock.npz"
    np.savez(starts, x=np.zeros(n), shard=shard, shard_id=np.array([7, 9])[:len(counts)],
             shard_generated=np.array(counts, dtype=int))
    zeros = np.zeros(n)
    np.savez(ref, xvd_front=zeros, yvd_front=zeros, zvd_front=zeros, pxvd_front=zeros, pyvd_front=zeros,
             pzvd_front=np.full(n, 40. if empty_arm == "stock" else 95.), tvd_front=zeros, time=zeros,
             isvd_front=np.ones(n, bool), ifile=shard)
    for j, indices in enumerate(chunks):
        idx = np.array(indices, dtype=int)
        m = idx.size
        np.savez(tmp_path / f"fast{j}.npz", idx=idx,
                 fate=np.full(m, DC.C.STOPPED if empty_arm == "fast" else DC.C.PLANE),
                 x=np.zeros((m, 3)), u=np.tile([0., 0., 1.], (m, 1)), T=np.full(m, 95.), t=np.zeros(m),
                 t0=np.zeros(m), nfoil=np.zeros(m), nipa=np.zeros(m), nopa=np.zeros(m), path=np.zeros(m), wall=0.1)
    return ref, starts


def _observables(geom, field, state):
    p = state["p"][state["ok"]]
    return dict(p=p, cos=np.full(p.size, 0.6), r=np.full(p.size, 400.), t=np.full(p.size, 1000.),
                rmax=np.full(p.size, 600.))


def test_complete_reordered_chunks_and_counts(tmp_path):
    _, starts = _inputs(tmp_path)
    f, generated = DC.fast_plane(list(tmp_path.glob("fast*.npz")), starts)
    assert generated == {7: 3, 9: 2}
    np.testing.assert_array_equal(f["idx"], [3, 0, 4, 2, 1])
    np.testing.assert_array_equal(f["shard"], [9, 7, 9, 7, 7])


@pytest.mark.parametrize("chunks, message", [
    (((0, 1), (3, 4)), "missing fast output start indices: expected 5, found 4"),
    (((0, 1, 2), (2, 3, 4)), "duplicate fast output start indices"),
    (((0, 1, 2), (3, 3)), "duplicate fast output start indices"),
    (((0, 1, 2), (3, 5)), "start index out of range"),
    (((0, 1, 2), (3, -1)), "start index out of range"),
    ((), "no fast output chunks"),
])
def test_reject_bad_chunk_coverage(tmp_path, chunks, message):
    _, starts = _inputs(tmp_path, chunks)
    with pytest.raises(ValueError, match=message):
        DC.fast_plane(list(tmp_path.glob("fast*.npz")), starts)


@pytest.mark.parametrize("ids, counts", [([7, 9], [3, 1]), ([7, 9], [2, 3]), ([7, 7], [3, 2]),
                                         ([7, 9], [3.5, 1.5]), ([7, 9], [6, -1])])
def test_reject_inconsistent_start_counts(tmp_path, ids, counts):
    _, starts = _inputs(tmp_path)
    np.savez(starts, x=np.zeros(5), shard=[7, 7, 7, 9, 9], shard_id=ids, shard_generated=counts)
    with pytest.raises(ValueError, match="starts shard_generated counts"):
        DC.fast_plane(list(tmp_path.glob("fast*.npz")), starts)


def test_zero_generated_shard_metadata(tmp_path):
    _, starts = _inputs(tmp_path)
    np.savez(starts, x=np.zeros(5), shard=[7, 7, 7, 9, 9], shard_id=[9, 12, 7], shard_generated=[2, 0, 3])
    _, generated = DC.fast_plane(list(tmp_path.glob("fast*.npz")), starts)
    assert generated == {9: 2, 12: 0, 7: 3}


@pytest.mark.parametrize("stock, fast, exposure, ratio", [([], [], [], None), ([0], [0], [0], None),
                                                         ([3], [2], [10], 2 / 3), ([3], [0], [10], 0.),
                                                         ([0, 0], [1, 0], [10, 10], None)])
def test_unavailable_ratio_uncertainty(stock, fast, exposure, ratio):
    with np.errstate(all="raise"):
        result = DC.ratio_by_shard(stock, fast, exposure)
    assert result["ratio"] == ratio
    assert result["ratio_err"] is None
    if not sum(exposure):
        assert result["y_stock"] is None and result["y_fast"] is None
        assert result["y_stock_err"] is None and result["y_fast_err"] is None
    json.dumps(result, allow_nan=False)


def test_zero_accepted_fast_yield_keeps_exposure():
    result = DC.ratio_by_shard([2, 3], [0, 0], [10, 20])
    assert result["generated"] == 30 and result["y_stock"] == 5 / 30
    assert result["fast"] == result["y_fast"] == result["ratio"] == 0
    assert result["ratio_err"] == 0.


@pytest.mark.parametrize("case", ["complete", "fast", "stock", "empty-heldout", "one-shard", "zero-generated"])
def test_compare_and_numbers_cli_edge_cases(tmp_path, monkeypatch, capsys, case):
    options = dict(empty_arm=case if case in ("fast", "stock") else None)
    if case == "one-shard":
        options.update(counts=(5,), chunks=((3, 0), (4, 2, 1)))
    elif case == "zero-generated":
        options.update(counts=(), chunks=((),))
    ref, starts = _inputs(tmp_path, **options)
    monkeypatch.setattr(DC.CC, "observables", _observables)
    monkeypatch.setattr(DC.C, "from_offline", lambda *args: None)
    monkeypatch.setattr(DC.C, "field_maps", lambda *args: None)
    dev = 10 if case == "empty-heldout" else 0
    report, numbers = tmp_path / "report.json", tmp_path / "numbers.json"
    with np.errstate(all="raise"):
        DC.main(["--ref", str(ref), "--starts", str(starts), "--fast", str(tmp_path / "fast*.npz"),
                 "--dev-shards", str(dev), "--out", str(report)])
    json.loads(capsys.readouterr().out)
    DC.main(["--numbers", str(report), str(numbers)])
    rep, nums = json.loads(report.read_text()), json.loads(numbers.read_text())
    json.dumps(rep, allow_nan=False)
    json.dumps(nums, allow_nan=False)
    y = nums["rmax_450"]["heldout"]
    assert nums["generated_per_arm"] == (0 if case == "zero-generated" else 5)
    if case in ("empty-heldout", "zero-generated"):
        assert y["generated"] == y["stock"] == y["fast"] == 0
        assert y["y_stock"] is None and y["y_fast"] is None and y["ratio_err"] is None
    else:
        assert y["generated"] == 5
        assert y["stock"] == (0 if case == "stock" else 5)
        assert y["fast"] == (0 if case == "fast" else 5)
        assert y["ratio"] == (None if case == "stock" else 0 if case == "fast" else 1)
        if case == "one-shard":
            assert y["ratio_err"] is None
    for v in nums["shapes_heldout"].values():
        if case in ("fast", "stock", "empty-heldout", "zero-generated"):
            assert v["chi2"] is None and v["mean_diff_err"] is None
            if case != "stock":
                assert v["mean_fast"] is None
            if case != "fast":
                assert v["mean_stock"] is None
    if case in ("empty-heldout", "one-shard", "zero-generated", "fast", "stock"):
        DC.main(["--plot", str(report), str(tmp_path / "plot.png")])
        assert (tmp_path / "plot.png").stat().st_size > 0


def test_flow_counts_keep_histogram_edges_and_window_statistics():
    values = {k: np.array([e[0] - 1., e[0], e[-1], e[-1] + 1.]) for k, e in DC.BINS.items()}
    for k, v in DC.shapes(values, values, 4).items():
        assert v["stock_flow"] == v["fast_flow"] == dict(under=1, over=1)
        assert sum(v["stock"]) == sum(v["fast"]) == 0.5
        assert v["edges"] == DC.BINS[k].tolist()
        assert v["chi2"] is None and v["chi2_ndf"] is None


@pytest.mark.parametrize("bad_field", ["idx", "fate"])
def test_reject_noninteger_indices_and_wrong_final_row_counts(tmp_path, bad_field):
    _, starts = _inputs(tmp_path, chunks=((0, 1, 2, 3, 4),))
    path = tmp_path / "fast0.npz"
    with np.load(path) as f:
        data = dict(f)
    data[bad_field] = data[bad_field].astype(float) if bad_field == "idx" else data[bad_field][:-1]
    np.savez(path, **data)
    with pytest.raises(ValueError, match="integer array" if bad_field == "idx" else "row counts"):
        DC.fast_plane([path], starts)


@pytest.mark.parametrize("counts, chunks", [((), ((),)), ((3, 2), ((0, 1, 2), (3, 4)))])
def test_empty_selection_with_real_observables(tmp_path, counts, chunks):
    ref, starts = _inputs(tmp_path, chunks=chunks, counts=counts, empty_arm="fast")
    geom = SimpleNamespace(beam=SimpleNamespace(axis_x=0.))
    field = lambda x: np.tile([0., 0., 1.], (len(x), 1))
    with np.errstate(all="raise"):
        rep = DC.compare(ref, str(tmp_path / "fast*.npz"), starts, geom, field, dev_shards=0)
    json.dumps(rep, allow_nan=False)
    result = DC.numbers(rep)
    assert result["rmax_450"]["all"]["generated"] == sum(counts)
    assert result["rmax_450"]["all"]["fast"] == 0
    assert result["shapes_heldout"]["p"]["mean_fast"] is None


def test_nonempty_histogram_statistics_regression():
    so = {k: np.repeat((e[:-1] + e[1:])[:2] / 2, [12, 8]) for k, e in DC.BINS.items()}
    fo = {k: np.repeat((e[:-1] + e[1:])[:2] / 2, [8, 16]) for k, e in DC.BINS.items()}
    for v in DC.shapes(so, fo, 100).values():
        assert v["chi2"] == pytest.approx(16 / 20 + 64 / 24)
        assert v["ndf"] == 2
        assert v["chi2_ndf"] == pytest.approx((16 / 20 + 64 / 24) / 2)
        assert v["shape_ndf"] == 1
        assert v["shape_chi2"] == pytest.approx(3.128888888888889)
        assert sum(v["stock"]) == pytest.approx(0.20)
        assert sum(v["fast"]) == pytest.approx(0.24)
