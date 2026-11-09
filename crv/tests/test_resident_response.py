"""response.resident: the same charges, photon count and rng state as the two-stage JAX twins (CPU JAX, synthetic tables)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")

from crvfastsim import chain, geometry, readout, runconfig  # noqa: E402
from crvfastsim.response import photons_jax, resident, sipm_jax  # noqa: E402
from test_readout_execution import _optical_table  # noqa: E402

SEED = 20261107


def setup(run="run1a_v01", events=3):
    rc = runconfig.get(run)
    names = list(rc.sectors)
    sector = np.arange(len(names), dtype=np.int64)[::-1].copy()
    n = len(sector)
    geom = geometry.CRVGeometry(
        names, sector, np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64),
        np.zeros((n, 3)), np.tile([9.64, 25.42, 1000.], (n, 1)), np.tile([0, 1, 2], (n, 1)))
    cfg = rc.chain_config(jax=True)
    cfg = replace(cfg, response=replace(cfg.response, sectors={
        s: replace(v, table=f"readout_execution_{abs(v.reflector)}") for s, v in rc.sectors.items()}))
    tabs = {f"readout_execution_{r}": photons_jax.DeviceTable(_optical_table(r))
            for r in {abs(v.reflector) for v in rc.sectors.values()}}
    m = n * events
    steps = pd.DataFrame({
        "subrun": np.ones(m, np.int64), "event": np.repeat(np.arange(1, events + 1), n),
        "bar": np.tile(np.arange(n), events), "edep": np.full(m, .004), "len": np.full(m, .3),
        "t0": np.full(m, 500.) + np.arange(m), "t1": np.full(m, 500.3) + np.arange(m),
        "x0": np.full(m, 2.), "x1": np.full(m, 2.2), "y0": np.zeros(m), "y1": np.full(m, .1),
        "z0": np.linspace(-300, 300, m), "z1": np.linspace(-300, 300, m) + .2,
        "px0": np.full(m, 1000.), "py0": np.zeros(m), "pz0": np.zeros(m), "p1": np.full(m, 1000.)})
    cond = runconfig.nominal_conditions(geom)
    cond.status[::7] = 1                                  # some channels masked (bit 0)
    return geom, cfg, tabs, steps, np.full(m, 13, np.int64), cond


def two_stage(geom, cfg, tabs, steps, pdg, cond, max_photons):
    rng, key = np.random.default_rng(SEED), jax.random.PRNGKey(SEED)
    k1, k2 = jax.random.split(key)
    s, c, t = photons_jax.make_photons(steps, pdg, geom, cfg.response, tabs, cond.deviation, rng, k1)
    keep = (readout.effective_status(cond)[c] & 0b101) == 0
    s, c, t = s[keep], c[keep], t[keep]
    label = chain.event_key(steps)[s] * chain.CHAN_MULT + c
    q = sipm_jax.simulate(label, t, s, np.ones((40, 40)), cfg.sipm, rng, k2, t_end=cfg.window[1],
                          max_photons=max_photons)
    return q, len(s), rng.random(4)


def resident_path(geom, cfg, tabs, steps, pdg, cond, max_photons):
    rng, key = np.random.default_rng(SEED), jax.random.PRNGKey(SEED)
    timers = {}
    q = resident.charges(steps, pdg, geom, cond, cfg, rng, key, tabs, np.ones((40, 40)),
                         readout.effective_status(cond), chain.event_key(steps), chain.CHAN_MULT, timers,
                         max_photons=max_photons)
    return q, timers["photons"], rng.random(4)


@pytest.mark.pr_params("8388608")
@pytest.mark.parametrize("max_photons", [1 << 23, 64])
def test_resident_equals_two_stage(max_photons):
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        args = setup()
        a, na, ra = two_stage(*args, max_photons)
        b, nb, rb = resident_path(*args, max_photons)
    assert len(a["chan"]) > 50 and na == nb
    for k in ("chan", "time", "charge", "pe", "photon"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        assert a[k].dtype == b[k].dtype, k
    np.testing.assert_array_equal(ra, rb)


def test_chain_charges_flag_routes_to_resident(monkeypatch):
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        geom, cfg, tabs, steps, pdg, cond = setup(events=1)
        seen = []
        real = resident.charges
        monkeypatch.setattr(resident, "charges", lambda *a, **k: seen.append(1) or real(*a, **k))
        off = chain.charges(steps, None, geom, cond, cfg, np.random.default_rng(1), key=jax.random.PRNGKey(1),
                            tabs=tabs, pmap=np.ones((40, 40)), pdg=pdg)
        assert not seen
        on = chain.charges(steps, None, geom, cond, replace(cfg, resident_response=True), np.random.default_rng(1),
                           key=jax.random.PRNGKey(1), tabs=tabs, pmap=np.ones((40, 40)), pdg=pdg)
        assert seen
    for k in off:
        np.testing.assert_array_equal(off[k], on[k], err_msg=k)
