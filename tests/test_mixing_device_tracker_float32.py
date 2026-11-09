"""float32 device tracker against the staged float32 digitiser (TRKFASTSIM_DEVICE_FLOAT=core, MU2EFS_DEVICE_FLOAT=float32).

The device float is fixed at import, so these tests run only in a float32 process; test_mixing_device_tracker.py
starts one. In a float64 process the module is skipped.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mu2efastsim.testing")
pytest.importorskip("trkfastsim")
from trkfastsim import devfloat as df

if not df.SINGLE:
    pytest.skip("tracker float32 mode is not active in this process", allow_module_level=True)

from test_mixing_device_tracker import x64, helpers, cond, collection, timing, evaluate  # noqa: F401,E402
from mixing.device_tracker import Capacities, prepare  # noqa: E402

# Voltage sums run over the padded cluster width, so a different width (the staged power-of-two buckets, or a tier)
# changes the float32 summation order. Digitised integers must match; analog times agree to float32 rounding of the
# per-straw relative time (observed <= 2e-6 ns; the TDC LSB is 19.5 ps).
ANALOG_NS = 1e-5


def _same(got, expected):
    for name in expected:
        if name == "t_analog":
            np.testing.assert_allclose(got[name], expected[name], rtol=0, atol=ANALOG_NS, err_msg=name)
        else:
            np.testing.assert_array_equal(got[name], expected[name], err_msg=name)


def _steps(c, helpers):
    return helpers._steps([helpers._step(c, 520), helpers._step(c, 520, t=40.), helpers._step(c, 521, ion=1),
                           helpers._step(c, 92, t=100.), helpers._step(c, 523, t=2800.),
                           helpers._step(c, 524, eion=1e-6), helpers._step(c, 525, t=900.)],
                          ev=[0, 0, 0, 2, 2, 2, 1])


@pytest.mark.pr_params("True")   # onspill is the mixing mode
@pytest.mark.parametrize("onspill", [False, True])
def test_float32_device_matches_staged_float32(cond, helpers, onspill):
    from trkfastsim.digi import digi_jax as reference
    from trkfastsim.digi.digicond import Conditions
    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = onspill
    st = _steps(c, helpers)
    times = timing([1700., 1675., 1695.] if onspill else [100000.] * 3, [10., -5., 20.])
    model = prepare(c, Capacities(64, 2048, 8, 512, 8, 2), 3)
    assert model.metadata["precision"] == "float32"
    got, errors = evaluate(model, collection(st, 8), times)
    assert not any(errors.values()), errors
    expected = reference.digitize(st, 3, c, jax.random.key(3),
                                  event_timing={"pbtime": times["mc_pbtime"], "event_length": times["event_length"]})
    assert len(got["sid"]) > 0
    _same(got, expected)


@pytest.mark.slow   # tiers per PR: the float64 test_long_step_and_end_tiers_digis_equal_wide
def test_float32_tiered_capacities_equal_wide(cond, helpers):
    from trkfastsim.digi.digicond import Conditions
    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = True
    st = _steps(c, helpers)
    times = timing([1700., 1675., 1695.], [10., -5., 20.])
    a, ae = evaluate(prepare(c, Capacities(64, 2048, 8, 512, 8, 2), 3), collection(st, 8), times)
    tiered = Capacities(2, 2048, 8, 4, 8, 2, long_steps=8, clusters_per_long_step=64, end_tiers=[(8, 16), (4, 512)])
    b, be = evaluate(prepare(c, tiered, 3), collection(st, 8), times)
    assert not any(ae.values()) and not any(be.values())
    assert len(a["sid"]) > 0
    _same(b, a)


def test_float32_flat_clusters_equal_dense(cond, helpers):
    """float32 roundoff is where a reshaped draw would show: the flat path gives the dense path's digis exactly."""
    from trkfastsim.digi.digicond import Conditions
    from test_mixing_device_tracker import FLAT, _flat_cluster_steps
    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = True
    st = _flat_cluster_steps(c, helpers)
    times = timing([1700., 1675., 1695.], [10., -5., 20.])
    dense = Capacities(8, 2048, 8, 4, 8, 2, long_steps=6, clusters_per_long_step=64, end_tiers=[(8, 16), (4, 512)])
    a, ae = evaluate(prepare(c, dense, 3), collection(st, 8), times)
    b, be = evaluate(prepare(c, Capacities(**{**vars(dense), **FLAT}), 3), collection(st, 8), times)
    assert not any(ae.values()) and not any(be.values()), (ae, be)
    assert len(a["sid"]) > 0
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_float32_flat_digit_entries_equal_pair_loop(cond, helpers):
    from trkfastsim.digi.digicond import Conditions
    from test_mixing_device_tracker import _flat_digit_steps  # float32 roundoff shows a reshaped sum
    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = True
    st = _flat_digit_steps(c, helpers)
    times = timing([1700., 1675., 1695.], [10., -5., 20.])
    a, ae = evaluate(prepare(c, Capacities(64, 1024, 8, 512, 8, 2), 3), collection(st, 8), times)
    b, be = evaluate(prepare(c, Capacities(64, 1024, 8, 512, 8, 2, digit_entries_per_straw=2), 3),
                     collection(st, 8), times)
    assert not any(ae.values()) and not any(be.values()), (ae, be)
    assert len(a["sid"]) > 0
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
