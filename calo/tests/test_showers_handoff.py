"""CE hand-off geometry and generated-denominator regression tests."""
from types import SimpleNamespace

import numpy as np

from calofastsim.showers.handoff import disk_front, from_cechain
from test_showers_response import geometry, source


def test_vacuum_front_and_time_of_flight():
    lib = geometry()
    x = np.array([[5., 0., 0.], [30., 0., 0.]])
    u = np.tile([0., 0., 1.], (2, 1))
    result = disk_front([0, 0, 0], lib, x, u, np.array([100., 100.]), np.array([10., 10.]))
    np.testing.assert_array_equal(result["event_row"], [0])
    np.testing.assert_array_equal(result["disk"], [0])
    np.testing.assert_allclose(result["position"], [[5, 0, 100]], atol=1e-8)
    beta = np.sqrt(100 * (100 + 2 * .51099895)) / (100 + .51099895)
    np.testing.assert_allclose(result["time"], [10 + 100 / (299.792458 * beta)], atol=1e-10)


def test_state_already_on_front_is_not_skipped():
    lib = geometry()
    result = disk_front([0, 0, 0], lib, np.array([[5., 0., 100.]]), np.array([[0., 0., 1.]]),
                        np.array([100.]), np.array([20.]))
    np.testing.assert_array_equal(result["disk"], [0])
    np.testing.assert_array_equal(result["time"], [20])


def test_from_cechain_keeps_generated_denominator():
    lib, _, _ = source()
    result = SimpleNamespace(fate=np.array([1, 2, 1]), x=np.array([[5., 0., 0.], [5., 0., 0.], [30., 0., 0.]]),
                             u=np.tile([0., 0., 1.], (3, 1)), T=np.full(3, 100.), t=np.full(3, 1000.))
    out = from_cechain(result, [0, 0, 0], lib, np.random.default_rng(0))
    assert out["generated_events"] == 3
    assert out["tracker_entry_events"] == 2
    assert out["disk_front_events"] == 1
    np.testing.assert_array_equal(out["entries"]["event_row"], [0])


def test_empty_cechain_input():
    lib, _, _ = source()
    result = SimpleNamespace(fate=np.array([2]), x=np.zeros((1, 3)), u=np.array([[0., 0., 1.]]), T=np.ones(1), t=np.zeros(1))
    out = from_cechain(result, [0, 0, 0], lib, np.random.default_rng(0))
    assert out["generated_events"] == 1 and out["disk_front_events"] == 0
    assert len(out["steps"]["entry"]) == 0
