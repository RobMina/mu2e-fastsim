"""Unit tests: bar geometry re-implementation and ray-box crossings (CPU JAX; needs only the Offline counters file)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import crossings as cx                                  # noqa: E402
from crvfastsim import geometry as geo                                  # noqa: E402


@pytest.fixture(scope="module")
def g():
    try:
        return geo.from_offline_config(geo.find_counters("Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"))
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_bar_count_and_first_bar(g):
    assert g.nbars == 576
    assert np.allclose(g.center[0], [-3904.0, 4387.0, 21439.55])
    assert np.allclose(g.halflen[0], [3000.0, 9.9, 25.65])


def test_ex_module_gap_and_stagger(g):
    ex = g.sector == 0
    z = g.center[ex & (g.layer == 0) & (g.module <= 1), 2]
    assert np.isclose(np.diff(np.sort(z)).max(), 51.3 + 4.433)
    assert np.isclose(g.center[g.bar_index(0, 0, 1, 0), 2] - g.center[g.bar_index(0, 0, 0, 0), 2], -42.0)


def test_vertical_track_crosses_four_layers_of_one_counter(g):
    x, _, z = g.center[g.bar_index(0, 1, 0, 5)]
    c = cx.crossings(np.array([[x, 6000.0, z + 21.0]]), np.array([[0.0, -1.0, 0.0]]), g)
    ex = c[g.sector[c[:, 1].astype(int)] == 0]
    assert len(ex) == 4
    assert np.allclose(ex[:, 4], 19.8)


def test_slanted_path_length(g):
    b = g.bar_index(0, 1, 0, 5)
    x, y, z = g.center[b]
    th = np.deg2rad(30.0)
    d = np.array([[np.sin(th), -np.cos(th), 0.0]])  # tilted along the bar length: no width clipping
    c = cx.crossings(np.array([[x - (6000 - y) * np.tan(th), 6000.0, z]]), d, g)
    hit = c[c[:, 1].astype(int) == b]
    assert len(hit) == 1
    assert np.isclose(hit[0, 4], 19.8 / np.cos(th), rtol=1e-9)


def test_numpy_and_jax_backends_agree(g):
    if not cx.HAVE_JAX:
        pytest.skip("no jax")
    rng = np.random.default_rng(5)
    o = np.c_[rng.uniform(-6000, 0, 2000), np.full(2000, 6000.0), rng.uniform(20000, 26000, 2000)]
    d = np.c_[rng.normal(0, 0.3, 2000), -np.ones(2000), rng.normal(0, 0.3, 2000)]
    d /= np.linalg.norm(d, axis=1)[:, None]
    a = cx.crossings(o, d, g, backend="numpy")
    b = cx.crossings(o, d, g, backend="jax")
    assert a.shape == b.shape and np.array_equal(a[:, :2], b[:, :2]) and np.allclose(a[:, 2:], b[:, 2:], rtol=0, atol=1e-9)
