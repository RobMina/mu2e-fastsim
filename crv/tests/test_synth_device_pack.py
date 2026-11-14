"""Waveform synthesis packed on the device (capacity mode): waveform_jax.gather_tw gives pack()'s [C, Q] matrices,
and _synth_jax_fixed gives the same digis from the resident response's device buffer as from an upload."""
import weakref

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from crvfastsim.capacity import Capacities  # noqa: E402
from crvfastsim.reco import cli, waveform, waveform_jax as WJ  # noqa: E402
from crvfastsim.response import resident  # noqa: E402

from test_capacity import _charges  # noqa: E402


@pytest.mark.parametrize("seed", [4, 9])
def test_gather_tw_equals_pack(seed):
    cfg = waveform.WaveformConfig()
    Q, tdc0 = _charges(150, seed)
    rows, t, q = Q["row"], Q["time"], Q["charge"]
    cid = waveform.charge_clusters(rows, t, cfg)
    first = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    nq = np.bincount(cid)
    qb = cli._p2(nq)
    dev = (jnp.asarray(t, dtype=jnp.float64), jnp.asarray(q, dtype=jnp.float64))
    for Qb in np.unique(qb):
        c = np.flatnonzero(qb == Qb)[::2]                                    # every other cluster of this Q
        idx = np.concatenate([np.arange(first[k], first[k] + nq[k]) for k in c])
        pk = WJ.pack(rows[idx], t[idx], q[idx], tdc0, cfg, Q=int(Qb))
        pad = 3
        T, W = WJ.gather_tw(*dev, np.r_[first[c], np.zeros(pad, np.int64)], np.r_[nq[c], np.zeros(pad, np.int64)],
                            cfg.single_pe_reference_charge, Q=int(Qb))
        T, W = np.asarray(T), np.asarray(W)
        assert T.dtype == pk["T"].dtype and W.dtype == pk["W"].dtype
        np.testing.assert_array_equal(T[:len(c)], pk["T"])
        np.testing.assert_array_equal(W[:len(c)], pk["W"])
        assert not T[len(c):].any() and not W[len(c):].any()
        np.testing.assert_array_equal(nq[c], pk["n"])


def test_resident_buffer_handle():
    rng = np.random.default_rng(2)
    pieces = []
    for width, n in ((400, 250), (400, 399)):
        cols = (rng.integers(0, 30, width), np.round(rng.uniform(0, 50, width), 1), rng.uniform(size=width),
                rng.uniform(size=width), rng.integers(0, 10 ** 6, width))
        pieces.append((tuple(jnp.asarray(x) for x in cols), n))
    out = resident._sorted_charges(pieces, Capacities(record=True))
    dev = resident.device_charges(out["time"])
    assert dev is not None
    n = len(out["time"])
    np.testing.assert_array_equal(np.asarray(dev[0], np.float64)[:n], out["time"])
    np.testing.assert_array_equal(np.asarray(dev[1], np.float64)[:n], out["charge"])
    assert resident.device_charges(out["time"].copy()) is None


@pytest.mark.parametrize("noise", [False, True])
def test_fixed_synthesis_same_from_device_buffer_and_upload(noise):
    cfg = waveform.WaveformConfig()
    Q, tdc0 = _charges(200, 5)
    nz = cfg.noise if noise else 0.0
    up = cli._synth_jax_fixed(WJ, Q, tdc0, cfg, Capacities(record=True), noise=nz, seed=3)
    size = 2 * len(Q["time"]) + 17                                           # a padded buffer, as the resident one
    buf = tuple(jnp.asarray(np.pad(Q[k], (0, size - len(Q[k]))), dtype=jnp.float64) for k in ("time", "charge"))
    resident._LAST[:] = [(weakref.ref(Q["time"]), *buf)]
    try:
        got = cli._synth_jax_fixed(WJ, Q, tdc0, cfg, Capacities(record=True), noise=nz, seed=3)
    finally:
        resident._LAST.clear()
    assert len(up["adc"]) > 300
    for k in up:
        assert up[k].dtype == got[k].dtype, k
        np.testing.assert_array_equal(up[k], got[k], err_msg=k)
