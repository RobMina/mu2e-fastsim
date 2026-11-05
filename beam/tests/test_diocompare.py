"""Unit tests: per-shard paired yield ratio and per-generated shapes of beamfastsim.diocompare."""
import numpy as np

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
