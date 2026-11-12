"""Device disk-front hand-off against the host adapter (showers.handoff.disk_front) in a uniform field."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
# mu2efastsim.eslab_jax is imported by disk_front at call time, after the test hooks (mu2efastsim.testing) load
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from calofastsim.showers.handoff import disk_front  # noqa: E402
from calofastsim.showers.handoff_device import disk_front as disk_front_device, disk_geometry  # noqa: E402
from test_device_chain import _geometry_keys, _grid  # noqa: E402


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_matches_host_in_uniform_field(dtype):
    rng = np.random.default_rng(2)
    library = _geometry_keys(*_grid())
    n = 400
    r, phi = rng.uniform(0, 700, n), rng.uniform(-np.pi, np.pi, n)
    x = np.column_stack((r * np.cos(phi), r * np.sin(phi), rng.uniform(-900, -100, n)))
    cos = rng.uniform(0.3, 0.9, n)
    a = rng.uniform(-np.pi, np.pi, n)
    u = np.column_stack((np.sqrt(1 - cos ** 2) * np.cos(a), np.sqrt(1 - cos ** 2) * np.sin(a), cos))
    T, t = rng.uniform(60, 104, n), rng.uniform(500, 1500, n)
    field = np.array([0.0, 0.0, 1.0])
    host = disk_front(field, library, x, u, T, t.copy())
    dev = disk_front_device(field, disk_geometry(library, dtype), jnp.asarray(x, dtype), jnp.asarray(u, dtype),
                            jnp.asarray(T, dtype), jnp.asarray(t, dtype), jnp.ones(n, bool))
    dev = {k: np.asarray(v) for k, v in dev.items()}
    found = np.flatnonzero(dev["found"])
    tol = 1e-6 if dtype is np.float64 else 0.05
    np.testing.assert_array_equal(found, host["event_row"])
    assert 0 < len(found) < n
    np.testing.assert_array_equal(dev["disk"][found], host["disk"])
    np.testing.assert_allclose(dev["position"][found], host["position"], atol=tol)
    np.testing.assert_allclose(dev["direction"][found], host["direction"], atol=tol / 100)
    np.testing.assert_allclose(dev["time"][found], host["time"], atol=1e-7 if dtype is np.float64 else 0.05)
    np.testing.assert_allclose(dev["radius"][found], host["radius"], atol=tol)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_inactive_slow_lanes_change_neither_results_nor_the_scan(dtype):
    # a very slow electron that is not active (e.g. one that stopped in the tracker) used to set the plane scan's
    # length for every step; the active lanes' results are the same with or without it, bit for bit
    rng = np.random.default_rng(4)
    library = _geometry_keys(*_grid())
    n = 64
    r, phi = rng.uniform(0, 700, n), rng.uniform(-np.pi, np.pi, n)
    x = np.column_stack((r * np.cos(phi), r * np.sin(phi), rng.uniform(-900, -100, n)))
    cos = rng.uniform(0.3, 0.9, n)
    a = rng.uniform(-np.pi, np.pi, n)
    u = np.column_stack((np.sqrt(1 - cos ** 2) * np.cos(a), np.sqrt(1 - cos ** 2) * np.sin(a), cos))
    T, t = rng.uniform(60, 104, n), rng.uniform(500, 1500, n)
    geo, field = disk_geometry(library, dtype), np.array([0.0, 0.0, 1.0])

    def run(x, u, T, t, active):
        out = disk_front_device(field, geo, jnp.asarray(x, dtype), jnp.asarray(u, dtype), jnp.asarray(T, dtype),
                                jnp.asarray(t, dtype), jnp.asarray(active))
        return {k: np.asarray(v) for k, v in out.items()}
    base = run(x, u, T, t, np.ones(n, bool))
    slow = run(np.vstack([x, x[:1]]), np.vstack([u, u[:1]]), np.append(T, 1e-9), np.append(t, t[0]),
               np.append(np.ones(n, bool), False))
    for k, v in base.items():
        w = slow[k] if np.ndim(v) == 0 else slow[k][:n]
        assert np.asarray(v).tobytes() == np.asarray(w).tobytes(), k
