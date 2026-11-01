"""Range-mean layer physics, shipped G4 tables, the single-foil emulator, gun lists and chain metrics."""
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mu2efastsim import eloss

from beamfastsim import g4ref, g4tables, geometry, layerphys, transport


@pytest.mark.parametrize("table,material", [("target", "StoppingTarget_Al"), ("slab", "G4_Al")])
def test_g4_tables_shipped_and_consistent(table, material):
    t = g4tables.load(g4tables.PATHS[table])
    assert t["meta"]["material"] == material and "11.3.2" in t["meta"]["release"]
    e = t["energy"]
    for q in ("mu-", "mu+"):
        r, s = t[f"csda_range_{q}"], t[f"dedx_nr_{q}"]
        assert np.all(np.diff(r) > 0) and np.all(s > 0)
        # CSDA range is the integral of 1/S from the first node
        i = np.searchsorted(e, [0.1, 1.0, 5.0])
        integral = r[0] + np.concatenate([[0], np.cumsum(0.5 * (1 / s[1:] + 1 / s[:-1]) * np.diff(e))])
        assert np.allclose(integral[i], r[i], rtol=2e-4)
    # mu- and mu+ differ below a few MeV (ICRU73QO vs Bragg, Barkas), agree at high energy
    k = np.searchsorted(e, [0.5, 30.0])
    d = t["dedx_nr_mu-"][k] / t["dedx_nr_mu+"][k] - 1
    assert abs(d[0]) > 1e-3 and abs(d[1]) < 3e-3          # target mu+ keeps a 0.2 MeV / T join term
    assert np.all(t["msc_lambda1"] > 0)


def test_target_and_slab_charge_contract():
    """mu- identical; mu+ differs only through the low-energy model, joined above 0.2 MeV as (1 + d * 0.2 / T)."""
    a, b = g4tables.load(g4tables.SLAB), g4tables.load(g4tables.TARGET)
    assert np.array_equal(a["energy"], b["energy"])
    for k in ("csda_range_mu-", "dedx_nr_mu-", "msc_lambda1"):
        assert np.array_equal(a[k], b[k])
    e = a["energy"]
    r = a["dedx_nr_mu+"] / b["dedx_nr_mu+"] - 1
    assert 0.02 < np.abs(r).max() < 0.025
    hi = e >= 0.21
    assert np.ptp(r[hi] * e[hi] / 0.2) < 1.5e-3


def test_range_inverse_roundtrip():
    tb = layerphys.g4_tables("mu-")
    t = jnp.asarray(np.geomspace(1e-5, 500, 400))
    back = layerphys.energy_of(tb, layerphys.range_of(tb, t))
    assert np.allclose(np.asarray(back), np.asarray(t), rtol=1e-9)
    fast = layerphys.fast_tables(geometry.STOPPING_TARGET_AL, 1.0)
    t = jnp.asarray(np.geomspace(0.01, 500, 400))
    assert np.allclose(np.asarray(layerphys.energy_of(fast, layerphys.range_of(fast, t))), np.asarray(t), rtol=1e-9)


def test_true_path_limits():
    tbs = (layerphys.g4_tables("mu-"),)
    kind = jnp.zeros(3, jnp.int32)
    t0 = jnp.asarray([20.0, 1.4, 0.05])
    z = jnp.asarray([0.2, 0.1056, 0.1056])
    t = np.asarray(layerphys.true_path(tbs, kind, t0, z))
    assert 1.0 < t[0] / 0.2 < 1.001                 # fast muon: negligible correction
    assert 1.01 < t[1] / 0.1056 < 1.2               # near range-out: percent-level detour
    r = float(layerphys.range_of(tbs[0], jnp.asarray(0.05)))
    assert t[2] > r                                 # cannot traverse: ranges out


def test_range_mean_without_fluctuation_spread_is_slice_independent():
    """With the Urban width suppressed (tiny tcut), range-mean slices give the range-table exit for any n."""
    tbs = (layerphys.g4_tables("mu-"),)
    p = jnp.full(4, float(np.sqrt(3.0 * (3.0 + 2 * eloss.MMU))))
    from mu2efastsim import eloss_jax
    mc = eloss_jax.MatConsts(*(jnp.full(4, v) for v in eloss_jax.consts(geometry.STOPPING_TARGET_AL)))
    kind = jnp.zeros(4, jnp.int32)
    out = []
    for n in (1, 4, 32):
        out.append(np.asarray(layerphys.slices(jax.random.key(0), tbs, kind, mc, p, jnp.full(4, 0.2), n, 1e-6,
                                               jax.random.fold_in)))
    exact = float(layerphys.energy_of(tbs[0], layerphys.range_of(tbs[0], jnp.asarray(3.0)) - 0.2))
    for o in out:
        assert np.allclose(o, exact, rtol=2e-3)


def test_presets_and_legacy_default():
    assert transport.Params() == transport.params("legacy")
    g = transport.params("g4")
    assert (g.mean, g.law, g.msc_path) == ("range", "g4", True)
    with pytest.raises(ValueError):
        transport.layer_tables(transport.Params(mean="range", law="nope"))


def test_gunfile(tmp_path):
    g = dict(x=np.array([-3904.0, -3900.5]), y=np.array([1.0, -2.0]), z=np.array([4179.0, 4179.0]),
             px=np.array([1.0, 2.0]), py=np.array([0.5, -0.5]), pz=np.array([40.0, 30.0]),
             time=np.array([100.0, 120.0]), pdg=np.array([13, -13]))
    p = tmp_path / "gun.txt"
    assert g4ref.write_gunfile(g, p, charge=1) == 2
    rows = [line.split() for line in p.read_text().splitlines()]
    assert all(len(r) == 12 and r[7] == "-13" for r in rows) and float(rows[1][3]) == 2.0
    g4ref.write_gunfile(g, p)
    assert [line.split()[7] for line in p.read_text().splitlines()] == ["13", "-13"]


