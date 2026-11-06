"""Profiling counters and fixed-shape chunks preserve the source realization."""
import numpy as np
import pytest

from beamfastsim.flash_profile import step_summary, compare_saved


def test_step_summary_is_explicit_bound():
    out = step_summary([11, 11, -11], [2, 10, 4], [1, 2, 0], [0, 0, 0])
    assert out["11"]["iteration_lower_bound"] == 10
    assert out["11"]["vacuum_steps"] == 12
    assert out["11"]["active_step_fraction_upper_bound"] == .75
    assert out["-11"]["slab_steps"] == 0


def test_saved_comparison(tmp_path):
    path = tmp_path / "births.npz"
    np.savez(path, state_T=np.array([1., 2.]), photon_valid=np.array([[True], [False]]),
             overflow=[0, 0], capped=[False, False], loopers=[False, False])
    result = dict(state=dict(T=np.array([1., 2.])), records=dict(valid=np.array([[True], [False]])),
                  overflow=np.array([0, 0]), capped=np.zeros(2, bool), loopers=np.zeros(2, bool))
    assert all(x["exact"] for x in compare_saved(result, path).values())
    result["state"]["T"][0] += 1.
    assert compare_saved(result, path)["state_T"]["max_abs_delta"] == 1.
    result["state"]["x"] = np.zeros((2, 3))
    missing = compare_saved(result, path)["state_x"]
    assert missing["missing_reference"] and not missing["exact"]


def test_fixed_shape_compiled_chunks_preserve_rng():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from beamfastsim import flash_navigator_jax as FN
    from test_flash_target_jax import _geom, _starts, KEY

    geom = _geom()
    x, u = _starts(4, z=5800.)
    init, run = FN._build(geom, np.dtype(np.float64), 1., 30000., False, True, True, True,
                          np.array([0., 0., 1.]), photon_capacity=16)
    state = init((), jnp.asarray(x), jnp.asarray(u), jnp.full(4, 100.), jnp.zeros(4),
                 jnp.zeros(4, jnp.int32), jnp.full(4, 2.5))
    compiled = run.lower((), KEY, state, jnp.int32(50), 4, 4).compile()
    nf, full = compiled((), KEY, state, jnp.int32(50))
    na, part = compiled((), KEY, state, jnp.int32(20))
    nb, split = compiled((), KEY, part, jnp.int32(30))
    assert int(nf) == int(na) + int(nb)
    for a, b in zip(jax.tree_util.tree_leaves(full), jax.tree_util.tree_leaves(split)):
        np.testing.assert_array_equal(a, b)
