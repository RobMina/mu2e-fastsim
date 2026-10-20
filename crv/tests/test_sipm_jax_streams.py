"""device_sipm: the short/long stream split gives the same charges as one table; a long-table overflow is retried."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _inputs():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from crvfastsim.response import sipm, sipm_jax
    cfg = sipm.SiPMConfig()
    rng = np.random.default_rng(3)
    pmap = np.zeros((40, 40))
    pmap[18:22, 18:22] = 1.0                       # a narrow map piles photons onto few pixels: long streams
    pmap[5:35, 5:35] += 1e-3
    n_per = np.r_[rng.integers(1, 40, 60), [900, 1500]]   # ordinary SiPMs plus two shower SiPMs
    slot = np.repeat(np.arange(len(n_per)), n_per)
    t = rng.uniform(0, 200, len(slot))
    n = len(slot)
    NB, Ms, Ntot, q_ct = sipm_jax.device_buckets(n, cfg)
    dead = np.zeros(sipm_jax.NP2, bool)
    for x, y in cfg.inactive:
        dead[x * sipm_jax.NPIX + y] = True
    args = (jnp.asarray(np.pad(slot, (0, NB - n))), jnp.asarray(np.pad(t, (0, NB - n))), jnp.arange(NB) < n,
            jax.random.PRNGKey(7), jnp.asarray(np.cumsum(pmap.ravel()) / pmap.sum()), pmap.shape[1],
            jnp.asarray(dead), q_ct, cfg.overvoltage, cfg.time_constant, cfg.p1, cfg.p2, cfg.capacitance,
            NB, *Ms, Ntot)
    return sipm_jax, cfg, args


def _run(sj, cfg, args, L, SL, LL):
    return sj.device_sipm(*args, L, SL, LL, cfg.n_pixels_x, cfg.n_pixels_y)


def test_split_equals_single_table():
    sj, cfg, args = _inputs()
    ref = _run(sj, cfg, args, 4096, 1, 1)          # every stream fits the short table
    assert int(ref["n_long_streams"]) == 0 and not bool(ref["overflow"])
    spl = _run(sj, cfg, args, 8, 256, 4096)        # long streams in their own table
    assert int(spl["n_long_streams"]) > 0 and int(spl["max_long_stream"]) > 16 and not bool(spl["overflow"])
    for k in ("fired", "charge", "pe", "time", "root"):
        assert np.array_equal(np.asarray(ref[k]), np.asarray(spl[k])), k


def test_long_overflow_is_flagged_and_retried():
    sj, cfg, args = _inputs()
    small = _run(sj, cfg, args, 8, 256, 16)
    assert bool(small["stream_overflow"]) and bool(small["overflow"])
    SL, LL = sj.long_buckets(small)
    assert SL >= int(small["n_long_streams"]) and LL >= int(small["max_long_stream"])
    ref = _run(sj, cfg, args, 8, 256, 4096)
    old = sj.LONG_LL
    try:
        sj.LONG_LL = 16                             # force the retry path of run_device_sipm
        r = sj.run_device_sipm(args, cfg)
    finally:
        sj.LONG_LL = old
    assert not bool(r["overflow"])
    for k in ("fired", "charge", "pe"):
        assert np.array_equal(np.asarray(ref[k]), np.asarray(r[k])), k
