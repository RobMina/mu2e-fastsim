"""The bounded source must expose every truncation rather than silently dropping births."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from mu2efastsim import eslab_jax as sj
from mu2efastsim import materials
from beamfastsim.flash_source_jax import slab_source, validate_complete

@pytest.fixture(scope="module")
def table():
    return sj.lane_tables([(materials.BY_NAME["G4_W"], 5.0, 0.0814923, 0.00770222, 0.5)], msc=False)


def initial(n=16, energy=105.0):
    return sj.lane_state(np.full(n, energy), np.tile([0., 0., 1.], (n, 1)), np.zeros(n), np.full(n, 5.0))


def test_records_reproduce_primary_emission_totals(table):
    state = initial()
    result = slab_source(jax.random.PRNGKey(2), table, np.zeros(16, np.int32), state,
                         capacity=64, max_steps=5000, msc=False)
    validate_complete(result)
    np.testing.assert_array_equal(result["count"], result["state"]["nph"] - state["nph"])
    np.testing.assert_array_equal(np.asarray(result["records"]["valid"]).sum(axis=1), result["count"])
    np.testing.assert_allclose(np.asarray(result["records"]["energy"]).sum(axis=1), result["state"]["eph"], atol=1e-10)
    assert np.asarray(result["count"]).sum() > 0


def test_truncation_fails_closed(table):
    result = slab_source(jax.random.PRNGKey(2), table, np.zeros(16, np.int32), initial(),
                         capacity=1, max_steps=1000, msc=False)
    assert np.asarray(result["overflow"]).sum() > 0
    with pytest.raises(ValueError, match="Incomplete"):
        validate_complete(result)
    capped = slab_source(jax.random.PRNGKey(2), table, np.zeros(16, np.int32), initial(),
                         capacity=1, max_steps=1, msc=False)
    assert np.asarray(capped["capped"]).any()
    with pytest.raises(ValueError, match="capped"):
        validate_complete(capped)


def test_inactive_parents_emit_nothing(table):
    result = slab_source(jax.random.PRNGKey(4), table, np.zeros(16, np.int32), initial(energy=0.0), msc=False)
    validate_complete(result)
    assert int(result["steps"]) == 0
    assert not np.asarray(result["records"]["valid"]).any()


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_invalid_capacity(table, capacity):
    with pytest.raises(ValueError, match="capacity"):
        slab_source(jax.random.PRNGKey(2), table, np.zeros(16, np.int32), initial(), capacity=capacity)
