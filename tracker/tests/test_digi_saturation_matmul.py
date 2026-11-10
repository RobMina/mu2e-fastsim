"""Opt-in saturation_matmul: the saturation response as a polyphase-Toeplitz matmul equals the per-cluster gather up to
summation order ("tensor"), in the flat digitisation and the pair-by-pair loop; "bf16" is an approximation."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from trkfastsim.digi import digi_jax as DJ
from test_digi import cond


def _args():
    """Two straws: a modest one with several pairs and a saturated one (as test_flat_entries_equal_pair_loop)."""
    t = np.array([[1000., 1010., 1500., 3000.], [1001., 1011., 1501., 3001.],
                  [1200., 1205., 1210., 1215.], [1201., 1206., 1211., 1216.]])
    q = np.array([[.5] * 4, [.5] * 4, [80.] * 4, [80.] * 4])
    wd = np.array([[400.] * 4, [600.] * 4, [300.] * 4, [700.] * 4])
    step = np.tile(np.arange(4), (4, 1))
    return tuple(map(jnp.asarray, (t, q, wd, step, np.array([4, 4, 4, 4]), np.array([520, 1000]),
                                   np.array([False, False]), np.array([520, 1000]), np.array([True, True]),
                                   np.array([.3, -.2]))))


def _compare(ref, got, adc_tol):
    assert not got["failed"] and not got["sat_overflow"]
    np.testing.assert_array_equal(got["keep"], ref["keep"])
    keep = ref["keep"]
    for name in ("tdc", "tot", "step", "t_analog"):          # saturation moves only the ADC words
        np.testing.assert_array_equal(got[name][keep], ref[name][keep], err_msg=name)
    d = np.abs(got["adc"][keep].astype(np.int64) - ref["adc"][keep])
    assert d.max() <= adc_tol, d.max()
    assert np.abs(got["pmp"][keep].astype(np.int64) - ref["pmp"][keep]).max() <= adc_tol
    return d


@pytest.mark.parametrize("entries", [None, 8])
def test_tensor_matmul_equals_gather(cond, entries):
    key = jax.random.key(4)
    kw = {} if entries is None else {"entries": entries}
    ref = jax.device_get(DJ._electronics_kernel(cond, 4, 4, 2, **kw)(key, *_args()))
    got = jax.device_get(DJ._electronics_kernel(cond, 4, 4, 2, **kw, saturation_matmul="tensor",
                                                saturation_matmul_min_cap=1)(key, *_args()))
    assert ref.get("saturated_ends", 1) > 0
    _compare(ref, got, adc_tol=1)


def test_bf16_matmul_is_close(cond):
    key = jax.random.key(4)
    ref = jax.device_get(DJ._electronics_kernel(cond, 4, 4, 2, entries=8)(key, *_args()))
    got = jax.device_get(DJ._electronics_kernel(cond, 4, 4, 2, entries=8, saturation_matmul="bf16",
                                                saturation_matmul_min_cap=1)(key, *_args()))
    _compare(ref, got, adc_tol=8)


def test_min_cap_keeps_gather(cond):
    """Below saturation_matmul_min_cap the lane width keeps the gather: outputs are byte-identical to the default."""
    key = jax.random.key(4)
    ref = jax.device_get(DJ._electronics_kernel(cond, 4, 4, 2, entries=8)(key, *_args()))
    got = jax.device_get(DJ._electronics_kernel(cond, 4, 4, 2, entries=8, saturation_matmul="tensor",
                                                saturation_matmul_min_cap=8)(key, *_args()))
    for name in ("keep", "tdc", "tot", "adc", "pmp", "step", "t_analog"):
        np.testing.assert_array_equal(got[name], ref[name], err_msg=name)


def test_bad_mode(cond):
    with pytest.raises(ValueError, match="saturation_matmul"):
        DJ._electronics_parts(cond, 4, 4, 2, saturation_matmul="fp16")
