"""SiPM photon-map pixel id by stock release: "truncate" (Offline v13_40_00 and later) vs "lrint" (up to v13_39_00)."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crvfastsim import runconfig  # noqa: E402
from crvfastsim.response import contract, sipm  # noqa: E402


def _map():
    pmap = np.zeros((40, 40))
    pmap[6:34, 6:34] = np.random.default_rng(1).random((28, 28))   # content in bins 6-33, as photonMap.root
    return pmap


def test_truncate_keeps_the_map_bin_and_lrint_shifts_half():
    pmap = _map()
    n = 200_000
    tx, ty = sipm.draw_pixels(pmap, n, np.random.default_rng(2), "truncate")
    lx, ly = sipm.draw_pixels(pmap, n, np.random.default_rng(2), "lrint")
    assert tx.min() == 6 and tx.max() == 33 and ty.min() == 6 and ty.max() == 33
    assert np.all(pmap[tx, ty] > 0)                       # every photon on a pixel the map has content for
    dx, dy = lx - tx, ly - ty                             # same draws: lrint is the bin or the next one
    assert set(np.unique(dx)) == {0, 1} and set(np.unique(dy)) == {0, 1}
    assert abs((dx == 0).mean() - 0.5) < 0.01 and abs(((dx == 0) & (dy == 0)).mean() - 0.25) < 0.01
    assert lx.max() == 34                                 # half of bin 33 lands on pixel 34, which has no content


def test_unknown_pixel_id_is_refused():
    with pytest.raises(ValueError, match="pixel_id"):
        sipm.draw_pixels(_map(), 10, np.random.default_rng(0), "round")
    with pytest.raises(ValueError, match="pixel_id"):
        contract.validate_sipm(sipm.SiPMConfig(pixel_id="round"))


def test_truncate_saturates_more_on_a_crowded_sipm():
    """Many photons at once on one SiPM: fewer distinct pixels under truncation, so fewer PE."""
    pmap = _map()
    cfg = sipm.SiPMConfig(cross_talk=0.0)
    n_sipm, n_ph = 400, 400
    chan = np.repeat(np.arange(n_sipm), n_ph)
    t = np.zeros(len(chan))
    pe = {}
    for pid in sipm.PIXEL_IDS:
        out = sipm.simulate(chan, t, np.arange(len(chan)), pmap, dataclasses.replace(cfg, pixel_id=pid),
                            np.random.default_rng(3))
        pe[pid] = out["pe"].sum() / n_sipm
    assert pe["truncate"] < pe["lrint"] * 0.99
    # a single photon per SiPM is unaffected (no pixel is hit twice)
    one = {pid: sipm.simulate(np.arange(50_000), np.zeros(50_000), np.arange(50_000), pmap,
                              dataclasses.replace(cfg, pixel_id=pid), np.random.default_rng(4))["pe"].sum()
           for pid in sipm.PIXEL_IDS}
    assert abs(one["truncate"] / one["lrint"] - 1) < 4 / np.sqrt(one["lrint"])


def test_run_configurations_follow_their_stock_release():
    assert runconfig.get("run1a_v01").chain_config().sipm.pixel_id == "truncate"   # stock reference v13_41_00
    assert runconfig.get("run1b_v01").chain_config().sipm.pixel_id == "truncate"
    assert runconfig.get("run1a_v01", window="offspill").chain_config().sipm.pixel_id == "truncate"
    assert runconfig.get("extracted_v04").chain_config().sipm.pixel_id == "lrint"
    assert runconfig.get("run2_v10").chain_config().sipm.pixel_id == "lrint"
    assert sipm.SiPMConfig().pixel_id == "lrint"
    # everything else in the SiPM configuration is the default one
    a = dataclasses.replace(runconfig.get("run1a_v01").chain_config().sipm, pixel_id="lrint")
    assert a == sipm.SiPMConfig()


def _device(pixel_id, pmap, slot, t, key, **cfg_kw):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from crvfastsim.response import sipm_jax as sj
    cfg = sipm.SiPMConfig(pixel_id=pixel_id, **cfg_kw)
    n = len(slot)
    NB, Ms, Ntot, q_ct = sj.device_buckets(n, cfg)
    dead = np.zeros(sj.NP2, bool)
    for x, y in cfg.inactive:
        dead[x * sj.NPIX + y] = True
    args = (jnp.asarray(np.pad(slot, (0, NB - n))), jnp.asarray(np.pad(t, (0, NB - n))), jnp.arange(NB) < n,
            jax.random.PRNGKey(key), jnp.asarray(np.cumsum(pmap.ravel()) / pmap.sum()), pmap.shape[1],
            jnp.asarray(dead), q_ct, cfg.overvoltage, cfg.time_constant, cfg.p1, cfg.p2, cfg.capacitance,
            NB, *Ms, Ntot)
    return sj.run_device_sipm(args, cfg), sj


def test_device_truncate_saturates_like_numpy_on_a_crowded_sipm():
    """Device twin: the truncated mapping crowds the photons as the numpy port does (cross talk off)."""
    pmap = _map()
    n_sipm, n_ph = 1000, 400
    slot = np.repeat(np.arange(n_sipm), n_ph)
    t = np.zeros(len(slot))
    pe = {}
    for pid in sipm.PIXEL_IDS:
        r, _ = _device(pid, pmap, slot, t, 9, cross_talk=0.0)
        pe[pid] = float(np.asarray(r["pe"])[np.asarray(r["fired"])].sum()) / n_sipm
        F = sipm.simulate(slot, t, np.arange(len(slot)), pmap, sipm.SiPMConfig(pixel_id=pid, cross_talk=0.0),
                          np.random.default_rng(7))
        ref = F["pe"].sum() / n_sipm
        assert abs(pe[pid] / ref - 1) < 0.015, (pid, pe[pid], ref)
    assert pe["truncate"] < pe["lrint"] * 0.99


def test_device_retry_keeps_the_mapping(monkeypatch):
    pytest.importorskip("jax")
    from crvfastsim.response import sipm_jax as sj
    seen = []

    def device(*args, pixel_override=None, **kw):
        seen.append(kw)
        return {"stream_overflow": len(seen) == 1, "n_long_streams": 2, "max_long_stream": 65}

    monkeypatch.setattr(sj, "device_sipm", device)
    sj.run_device_sipm((), sipm.SiPMConfig(pixel_id="truncate"))
    assert seen == [{"truncate": True}, {"truncate": True}]
    seen.clear()
    sj.run_device_sipm((), sipm.SiPMConfig())
    assert seen == [{}, {}]                               # lrint: the call of main, no new keyword
    with pytest.raises(ValueError, match="pixel_id"):
        sj.run_device_sipm((), sipm.SiPMConfig(pixel_id="round"))


@pytest.mark.slow   # statistical (3,000 SiPMs)
def test_device_truncate_matches_numpy_truncate_statistically():
    jax = pytest.importorskip("jax")
    from crvfastsim.response import sipm_jax
    rng = np.random.default_rng(6)
    pmap = _map()
    cfg = sipm.SiPMConfig(pixel_id="truncate")
    n = rng.poisson(120, 3000)
    chan = np.repeat(np.arange(3000), n)
    t = np.repeat(rng.uniform(0, 1e4, 3000), n) + rng.normal(0, 3, len(chan))
    F = sipm.simulate(chan, t, np.arange(len(chan)), pmap, cfg, np.random.default_rng(1))
    J = sipm_jax.simulate(chan, t, np.arange(len(chan)), pmap, cfg, np.random.default_rng(2), jax.random.PRNGKey(2))
    L = sipm.simulate(chan, t, np.arange(len(chan)), pmap, sipm.SiPMConfig(), np.random.default_rng(1))
    assert abs(J["pe"].sum() / F["pe"].sum() - 1) < 4 / np.sqrt(len(F["pe"]))
    assert F["pe"].sum() < L["pe"].sum()                  # the crowding difference survives at 120 photons per SiPM
