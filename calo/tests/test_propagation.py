"""Synthetic table-edge tests, not validation with stock photon tables."""
import numpy as np
import pytest

from calofastsim.propagation import build_cdf, propagation_time, validate_table


def test_sequential_float32_cdf():
    contents = np.array([[0, 2, 3], [0, 0, 0]], dtype=np.float32)
    expected = np.array([[1e-6, 2.000001, 5.000001], [1e-6]*3], dtype=np.float32)
    expected /= expected[:, -1:]
    np.testing.assert_array_equal(build_cdf(contents), expected)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_lower_bound_edges_and_depth_clamping(backend):
    fn = propagation_time
    if backend == "jax":
        jax = pytest.importorskip("jax")
        from calofastsim.propagation_jax import propagation_time as fn
        fn = jax.jit(fn)
    cdf = np.array([[0.25, 0.5, 1], [1, 1, 1]], dtype=np.float32)
    times = [-1, 2, 5]
    depth = [-100, -0.5, 0, 1.99, 2, 100]
    u = [0, 0.25, np.nextafter(np.float32(.25), np.float32(1)), 1, 1, .5]
    np.testing.assert_array_equal(fn(depth, u, cdf, times, 2), [-1, -1, 2, 5, -1, -1])
    assert fn(np.empty(0), np.empty(0), cdf, times, 2).shape == (0,)
    assert float(fn(0, .5, cdf, times, 2)) == 2


@pytest.mark.parametrize("bins", [1, 2, 3, 4, 7, 8, 9])
def test_search_agrees_with_numpy_searchsorted(bins):
    cdf = build_cdf(np.arange(1, bins + 1)[None, :])
    draws = np.linspace(0, 1, 1001, dtype=np.float32)
    times = np.arange(bins, dtype=np.float32)
    expected = times[np.searchsorted(cdf[0], draws, side="left")]
    np.testing.assert_array_equal(propagation_time(0, draws, cdf, times, 1), expected)


@pytest.mark.parametrize("contents", [[], [[-1]], [[np.nan]], [[np.inf]]])
def test_invalid_contents(contents):
    with pytest.raises(ValueError):
        build_cdf(contents)


@pytest.mark.parametrize("depth,dz", [(np.finfo(np.float32).max, 1), (1, 1e-40),
                                    (np.float32(2**31), 1), (-np.finfo(np.float32).max, 1)])
def test_stock_undefined_depth_conversion_rejected(depth, dz):
    with pytest.raises(ValueError, match="signed int32"):
        propagation_time(depth, .5, [[1]], [2], dz)


def test_invalid_lookup():
    with pytest.raises(ValueError):
        validate_table([[.5, .4]], [1, 2], 1)
    with pytest.raises(ValueError):
        propagation_time(0, 1.01, [[1]], [2], 1)
