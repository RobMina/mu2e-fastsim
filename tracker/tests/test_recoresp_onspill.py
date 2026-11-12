"""The opt-in OnSpill response table (recoresp NAMED 'onspill') and the builder's entry-time window options."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.recoresp import table as RT


def test_onspill_table_loads_with_its_own_manifest():
    tab = RT.load(RT.resolve("onspill"))
    from mu2efastsim.library import store as LS
    lib = LS.load(RT.resolve("onspill"))
    assert lib.name == "tracker.response.ce_loophelix.onspill_pools.run1a"
    assert "pool_res" in tab and "time_eff" in tab and list(tab["time_plateau"]) == [500.0, 1400.0]
    e, eff = np.asarray(tab["time_edges"]), np.asarray(tab["time_eff"])
    lo = e[:-1]
    assert eff[lo < 350].max() < 0.01                       # no tracks before the on-spill digitization window
    assert eff[(lo >= 500) & (lo < 1400)].min() > 0.95      # the plateau
    assert eff[(lo >= 1700) & (lo < 2000)].max() < 0.2      # the window end, before the next microbunch
    assert np.all((eff >= 0) & (eff <= 1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_onspill_table_device_copy(dtype):
    pytest.importorskip("jax")
    from trkfastsim.recoresp import device as RD
    tb = RD.to_device(RT.load(RT.resolve("onspill")), dtype)
    assert tb.time_eff is not None and tb.pool_res is not None
    assert np.dtype(tb.time_eff.dtype) == np.dtype(dtype)


def _events(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2500, n)
    fit = rng.random(n) < np.where((t > 400) & (t < 1600), 0.9, 0.0)
    return pd.DataFrame(dict(has_entry=True, nrow=rng.integers(15, 60, n), p_true=104.0, tandip_t=rng.uniform(0.5, 1.1, n),
                             d0_t=50.0, rmax_t=550.0, t_t=t, fit=fit, p_reco=np.where(fit, 103.5, np.nan),
                             pz=np.where(fit, 60.0, np.nan), tandip=rng.uniform(0.5, 1.1, n), d0=50.0, rmax=550.0,
                             pdg=11.0, fitcon=0.5, chi2ndf=1.0, nactive=30.0, t0=t + 5))


def test_build_default_window_is_unchanged():
    d = _events()
    a = RT.build(d)
    b = RT.build(d, time_edges=RT.TIME_EDGES, plateau=(RT.T_PLATEAU, np.inf))
    assert set(a) == set(b) and "time_plateau" not in a
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    assert a["time_eff"][-1] == 1.0


def test_build_custom_window():
    d = _events()
    edges = (0.0, 400.0, 500.0, 1400.0, 1600.0, 1e9)
    tab = RT.build(d, time_edges=edges, plateau=(500.0, 1400.0))
    assert list(tab["time_plateau"]) == [500.0, 1400.0]
    assert tab["time_eff"][0] == 0.0 and tab["time_eff"][-1] == 0.0
    assert tab["time_eff"][2] == pytest.approx(1.0)
