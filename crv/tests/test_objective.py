"""crvfastsim.objective: the size-free shape term does not grow with the sample size, is ~1 for identical shapes, and
the module imports with numpy/awkward only (the Mu2eBO / grid contract)."""
import numpy as np

from crvfastsim.objective import crvresponse as R


def _hist(n, p, rng):
    return rng.multinomial(n, p).astype(float)


def test_size_free_at_fixed_difference():
    rng = np.random.default_rng(1)
    x = np.linspace(-3, 3, 41)
    p = np.exp(-0.5 * x ** 2); p /= p.sum()
    q = np.exp(-0.5 * (x - 0.3) ** 2); q /= q.sum()
    ref = _hist(5_000_000, q, rng)
    small = R.shape_term(_hist(20_000, p, rng), ref)
    big = R.shape_term(_hist(1_000_000, p, rng), ref)
    assert big["raw_chi2ndf"] > 10 * small["raw_chi2ndf"]          # the old form scales with N
    assert abs(big["chi2ndf"] - small["chi2ndf"]) < 4 * small["sigma"]
    assert big["sigma"] < small["sigma"]


def test_identical_shapes_near_one():
    rng = np.random.default_rng(2)
    p = np.full(30, 1 / 30)
    t = [R.shape_term(_hist(50_000, p, rng), _hist(2_000_000, p, rng))["chi2ndf"] for _ in range(20)]
    assert abs(np.mean(t) - 1.0) < 0.5


def test_objective_state_roundtrip():
    S = R.new_state()
    S["events"] = 1
    out = R.objective(S, R.new_state())
    assert out["n_terms"] == 0
