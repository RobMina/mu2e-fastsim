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
    spl = _run(sj, cfg, args, 8, 256, 4096)        # long streams in their own table
    assert int(spl["n_long_streams"]) > 0 and int(spl["max_long_stream"]) > 16 and not bool(spl["overflow"])
    L = 1 << (int(spl["max_long_stream"]) - 1).bit_length()   # the short table [hits, L] sized to the longest stream
    ref = _run(sj, cfg, args, L, 1, 1)             # every stream fits the short table
    assert int(ref["n_long_streams"]) == 0 and not bool(ref["overflow"])
    for k in ("fired", "charge", "pe", "time", "root"):
        assert np.array_equal(np.asarray(ref[k]), np.asarray(spl[k])), k


@pytest.mark.slow
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


def test_radix_lexsort_is_lexsort():
    """sipm_jax.lexsort(sort="radix") gives jnp.lexsort's permutation: ties, -0/+0, NaN, inf, bools, signed and
    unsigned integers, float32 and float64."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from crvfastsim.response import sipm_jax
    rng = np.random.default_rng(11)
    n = 4000
    for ftype in (np.float32, np.float64):
        t = rng.choice(np.array([0.0, -0.0, 1.5, -1.5, np.inf, -np.inf, np.nan, -np.nan, 2.25], ftype), n)
        t[: n // 2] = rng.normal(size=n // 2).astype(ftype)
        keys = (jnp.asarray(t), jnp.asarray(rng.integers(-3, 4, n)), jnp.asarray(rng.random(n) < 0.5),
                jnp.asarray(rng.integers(0, 5, n).astype(np.uint32)), jnp.asarray(rng.integers(-2**40, 2**40, n)))
        for k in range(1, len(keys) + 1):
            expected = np.asarray(jnp.lexsort(keys[:k]))
            np.testing.assert_array_equal(np.asarray(jax.jit(lambda *a: sipm_jax.lexsort(a, "radix"))(*keys[:k])),
                                          expected, err_msg=f"{ftype.__name__} {k} keys")
    with pytest.raises(ValueError, match="sort"):
        sipm_jax.lexsort(keys, "bitonic")


def test_radix_sort_same_charges():
    sj, cfg, args = _inputs()
    a = sj.device_sipm(*args, 8, 256, 4096, cfg.n_pixels_x, cfg.n_pixels_y)
    b = sj.device_sipm(*args, 8, 256, 4096, cfg.n_pixels_x, cfg.n_pixels_y, sort="radix")
    for name in ("slot", "time", "root", "fired", "charge", "pe", "overflow"):
        np.testing.assert_array_equal(np.asarray(a[name]), np.asarray(b[name]), err_msg=name)
    assert np.asarray(a["fired"]).sum() > 0
