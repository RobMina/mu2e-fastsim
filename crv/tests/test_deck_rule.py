"""The deck flag rule on stephist accumulators: variance choice per key, clustering factor, floor threshold."""
import numpy as np
import pytest

from valhist import deck_rule as R
from valhist import stephist as H

pytestmark = pytest.mark.slow   # campaign and validation tooling, not the simulation chain: runs with --runslow


def _acc(n, w2=None):
    n = np.asarray(n, float)
    return {"all": (n, n.copy() if w2 is None else np.asarray(w2, float))}


def _nstep(counts_at_centres):
    nb = H.BOOK["NStep"][0] + 2
    v = np.zeros(nb)
    for centre, count in counts_at_centres.items():
        v[1 + int(np.floor((centre + 9.5) / 10.0))] = count
    return v


def test_clustering_factor_uses_bin_centres_above_zero():
    v = _nstep({-4.5: 1000, 5.5: 3, 15.5: 1})
    assert R.clustering_factor(v) == pytest.approx((3 * 5.5 ** 2 + 15.5 ** 2) / (3 * 5.5 + 15.5))


def test_variance_kind_per_key():
    assert [R.variance_kind(k) for k in ("t", "t2", "lt", "posx", "EX/y", "bar", "E")] == \
        ["clustered_k", "clustered_k", "clustered_k", "exact", "exact", "poisson", "poisson"]


def test_compare_applies_k_exact_and_floor():
    nb = 102
    a, b = np.zeros(nb), np.zeros(nb)
    a[1:4], b[1:4] = [100, 100, 100], [130, 70, 100]
    ns = _nstep({5.5: 10, 25.5: 10})
    k = R.clustering_factor(ns)
    ref = {"t": _acc(a), "bar": _acc(a), "posx": _acc(a, 4 * a), "NStep": _acc(ns), "_total/all": _acc([1.0])}
    cand = {"t": _acc(b), "bar": _acc(b), "posx": _acc(b, 4 * b), "NStep": _acc(ns), "_total/all": _acc([1.0])}
    poisson = (900 / 230 + 900 / 170) / 3
    res = R.compare(ref, cand)
    assert "_total/all" not in res
    assert res["bar"]["chi2ndf"] == pytest.approx(poisson) and res["bar"]["flag"]
    assert res["t"]["chi2ndf"] == pytest.approx(poisson / k) and res["t"]["k"] == [k, k]
    assert res["posx"]["chi2ndf"] == pytest.approx(poisson / 4)
    floor = R.compare(ref, cand, ref, {**cand, "bar": _acc(a + np.r_[0, 40, -40, 0, np.zeros(nb - 4)])})
    assert floor["bar"]["floor_chi2ndf"] == pytest.approx((1600 / 240 + 1600 / 160) / 3)
    assert floor["bar"]["threshold"] == pytest.approx(3 * floor["bar"]["floor_chi2ndf"])
    assert not floor["bar"]["flag"]


def _hist(rng, n, scale=1.0):
    v = np.zeros(102)
    v[1:1 + n] = np.rint(rng.uniform(50, 150, n) * scale)
    return v


def _arm(rng, scale=1.0):
    """Accumulators for every key kind: k-factor (t, t2, lt), exact (posx, y/z), Poisson (bar, E with first-bin cut)."""
    acc = {k: _acc(_hist(rng, 20, scale)) for k in ("bar", "t", "t2", "lt", "E")}
    for k in ("posx", "EX/y"):
        n = _hist(rng, 20, scale)
        acc[k] = _acc(n, 3 * n)
    acc["NStep"] = _acc(_nstep({5.5: 100, 15.5: 30, 35.5: 5}))
    acc["_total/all"] = _acc([float(acc["bar"]["all"][0].sum())])
    return acc


def test_every_branch_with_and_without_floor():
    rng = np.random.default_rng(1)
    ref, cand = _arm(rng), _arm(rng)
    res = R.compare(ref, cand)
    assert set(res) == {"bar", "t", "t2", "lt", "E", "posx", "EX/y", "NStep"}
    k = R.clustering_factor(ref["NStep"]["all"][0])
    for key in ("bar", "E", "NStep"):
        n1, n2 = ref[key]["all"][0], cand[key]["all"][0]
        a, b = (n1[2:-1], n2[2:-1]) if key == "E" else (n1[1:-1], n2[1:-1])   # E: first booked bin left out
        m = (a + b) > 0
        assert res[key]["variance"] == "poisson"
        assert res[key]["chi2ndf"] == pytest.approx(((a - b)[m] ** 2 / (a + b)[m]).sum() / m.sum())
    for key in ("t", "t2", "lt"):
        n1, n2 = ref[key]["all"][0][1:-1], cand[key]["all"][0][1:-1]
        m = (n1 + n2) > 0
        assert res[key]["chi2ndf"] == pytest.approx(((n1 - n2)[m] ** 2 / (k * (n1 + n2))[m]).sum() / m.sum())
    for key in ("posx", "EX/y"):
        n1, n2 = ref[key]["all"][0][1:-1], cand[key]["all"][0][1:-1]
        m = (n1 + n2) > 0
        assert res[key]["variance"] == "exact"
        assert res[key]["chi2ndf"] == pytest.approx(((n1 - n2)[m] ** 2 / (3 * (n1 + n2))[m]).sum() / m.sum())
    assert all(v["threshold"] == 2.0 and v["floor_chi2ndf"] is None for v in res.values())
    assert all(v["flag"] == (v["chi2ndf"] > 2.0) for v in res.values())
    floored = R.compare(ref, cand, ref, _arm(rng))
    for key, v in floored.items():
        assert v["floor_chi2ndf"] is not None
        assert v["threshold"] == pytest.approx(max(2.0, 3.0 * v["floor_chi2ndf"]))
        assert v["flag"] == (v["chi2ndf"] > v["threshold"])


def test_summaries_are_summed_and_overlaps_refused(tmp_path):
    import json
    from valhist import stephist as H
    rng = np.random.default_rng(2)
    parts = []
    for i, shards in enumerate(([22, 24], [26])):
        hists = {a: H.to_json(_arm(rng)) for a in ("numpy", "jax")}
        p = tmp_path / f"s{i}.json"
        p.write_text(json.dumps({"shards": shards, "hists": hists}))
        parts.append((p, hists))
    arms, shards = R.from_summaries([p for p, _ in parts], ("numpy", "jax"))
    assert shards == [22, 24, 26]
    np.testing.assert_array_equal(arms["numpy"]["bar"]["all"][0], sum(np.asarray(h["numpy"]["bar"]["all"]["n"])
                                                                     for _, h in parts))
    with pytest.raises(ValueError, match="overlap"):
        R.from_summaries([parts[0][0], parts[0][0]], ("numpy",))
