"""recoresp: reco response tables (NumPy) and their device application (synthetic reference sample, no data files)."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.recoresp import table as RT


def test_entry_truth_of_a_known_circle():
    # e- in +Bz, at (axis + 300, 0) moving +y: centre on the -x side at distance R; r_c = |300 - R|
    bz, pt = 1.0, 60.0
    R = pt / (RT.KB * bz)
    x = np.array([[-3904.0 + 300.0, 0.0, 0.0]])
    p = np.array([[0.0, pt, 60.0]])
    pm, td, d0, rmax = RT.entry_truth(x, p, (-3904.0, 0.0), bz)
    np.testing.assert_allclose(pm, np.hypot(pt, 60.0))
    np.testing.assert_allclose(td, 60.0 / pt)
    np.testing.assert_allclose(d0, abs(abs(300.0 - R) - R))
    np.testing.assert_allclose(rmax, abs(300.0 - R) + R)


def _reference(n, rng):
    """Synthetic reference events: fit probability rising with nrow, residuals Gaussian with a low tail."""
    nrow = rng.integers(0, 90, n)
    td = rng.uniform(0.5, 1.2, n)
    pt = 104.0 / np.sqrt(1 + td ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = np.c_[-3904.0 + rng.uniform(380, 600, n) * np.cos(phi), rng.uniform(380, 600, n) * np.sin(phi), np.zeros(n)]
    p = np.c_[pt * np.cos(phi + 1.3), pt * np.sin(phi + 1.3), pt * td]
    ptrue, tdt, d0, rmax = RT.entry_truth(x, p, (-3904.0, 0.0), 1.0)
    fit = rng.uniform(size=n) < np.clip((nrow - 15) / 15, 0, 0.95)
    dp = np.where(rng.uniform(size=n) < 0.05, -rng.exponential(1.5, n), rng.normal(-0.05, 0.2, n))
    d = pd.DataFrame(dict(has_entry=True, nrow=nrow, p_true=ptrue, tandip_t=tdt, d0_t=d0, rmax_t=rmax, fit=fit,
                          e_x=x[:, 0], e_y=x[:, 1], e_z=x[:, 2], e_px=p[:, 0], e_py=p[:, 1], e_pz=p[:, 2]))
    d["p_reco"] = np.where(fit, ptrue + dp, np.nan)
    d["pz"] = np.where(fit, p[:, 2], np.nan)
    d["tandip"] = np.where(fit, tdt + rng.normal(0, 0.01, n), np.nan)
    d["d0"] = np.where(fit, np.abs(d0 + rng.normal(0, 10, n)), np.nan)
    d["rmax"] = np.where(fit, rmax + rng.normal(-3, 8, n), np.nan)
    d["pdg"] = np.where(fit, 11.0, np.nan)
    d["fitcon"] = np.where(fit, rng.uniform(size=n), np.nan)
    d["chi2ndf"] = np.where(fit, rng.gamma(20, 0.05, n), np.nan)
    d["nactive"] = np.where(fit, nrow * 0.9, np.nan)
    return d


def test_tables_reproduce_the_reference_efficiencies_and_residuals():
    rng = np.random.default_rng(1)
    d = _reference(40000, rng)
    tab = RT.build(d)
    v = _reference(20000, np.random.default_rng(2))
    o = RT.apply_numpy(tab, np.random.default_rng(3), v.has_entry.to_numpy(), v.nrow.to_numpy(),
                       v[["e_x", "e_y", "e_z"]].to_numpy(), v[["e_px", "e_py", "e_pz"]].to_numpy())
    assert abs(o["fit"].mean() - v.fit.mean()) < 0.01
    ref_ce = RT.CE_CUTS
    qual = v.fit & (v.fitcon > ref_ce["fitcon"])
    ce = qual & v.tandip.between(*ref_ce["tandip"]) & v.d0.between(*ref_ce["d0"]) & v.rmax.between(*ref_ce["rmax"])
    assert abs(o["ce"].mean() - ce.mean()) < 0.012
    res = (o["p_reco"] - o["p_true"])[o["qual"]]
    rr = (v.p_reco - v.p_true)[qual].to_numpy()
    assert abs(np.median(res) - np.median(rr)) < 0.02
    assert abs(np.mean(res < -1.0) - np.mean(rr < -1.0)) < 0.01
    assert not np.isfinite(o["p_reco"][~o["qual"]]).any()


def test_device_apply_matches_the_numpy_response():
    jax = pytest.importorskip("jax")
    from trkfastsim.recoresp import device as RD
    d = _reference(20000, np.random.default_rng(4))
    tab = RT.build(d)
    v = _reference(20000, np.random.default_rng(5))
    x, p = v[["e_x", "e_y", "e_z"]].to_numpy(), v[["e_px", "e_py", "e_pz"]].to_numpy()
    on = RT.apply_numpy(tab, np.random.default_rng(6), v.has_entry.to_numpy(), v.nrow.to_numpy(), x, p)
    od = RD.apply(RD.to_device(tab), jax.random.PRNGKey(7), jax.numpy.asarray(v.has_entry.to_numpy()),
                  jax.numpy.asarray(v.nrow.to_numpy()), jax.numpy.asarray(x), jax.numpy.asarray(p))
    od = {k: np.asarray(a) for k, a in od.items()}
    np.testing.assert_allclose(od["p_true"], on["p_true"], rtol=1e-12)
    for k in ("fit", "qual", "ce"):
        assert abs(od[k].mean() - on[k].mean()) < 0.012, k
    a, b = od["p_reco"][od["qual"]] - od["p_true"][od["qual"]], on["p_reco"][on["qual"]] - on["p_true"][on["qual"]]
    assert abs(np.median(a) - np.median(b)) < 0.02
    assert abs(np.quantile(a, 0.05) - np.quantile(b, 0.05)) < 0.1


def test_t0_is_drawn_on_its_own_stream_when_the_table_has_it():
    jax = pytest.importorskip("jax")
    from trkfastsim.recoresp import device as RD
    rng = np.random.default_rng(8)
    d = _reference(20000, rng)
    d["t_t"] = rng.uniform(500, 1500, len(d))
    d["t0"] = np.where(d.fit, d.t_t + rng.normal(1.2, 0.4, len(d)), np.nan)
    tab = RT.build(d)
    assert "icdf_dt0" in tab
    v = _reference(5000, np.random.default_rng(9))
    x, p = v[["e_x", "e_y", "e_z"]].to_numpy(), v[["e_px", "e_py", "e_pz"]].to_numpy()
    t = np.full(len(v), 800.0)
    tb = RD.to_device(tab)
    args = (jax.numpy.asarray(v.has_entry.to_numpy()), jax.numpy.asarray(v.nrow.to_numpy()), jax.numpy.asarray(x),
            jax.numpy.asarray(p))
    a = RD.apply(tb, jax.random.PRNGKey(1), *args)
    b = RD.apply(tb, jax.random.PRNGKey(1), *args, jax.numpy.asarray(t))
    for k in a:                                    # the other draws do not change when t0 is added
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True), k
    q = np.asarray(b["qual"])
    assert abs(np.median(np.asarray(b["t0"])[q] - 800.0) - 1.2) < 0.05
    assert not np.isfinite(np.asarray(b["t0"])[~q]).any()


def test_entry_time_turn_on_removes_tracks_before_the_window():
    jax = pytest.importorskip("jax")
    from trkfastsim.recoresp import device as RD
    rng = np.random.default_rng(10)
    d = _reference(30000, rng)
    d["t_t"] = rng.uniform(100, 1100, len(d))
    d["fit"] = d.fit & (d.t_t > 210)                     # no reconstruction before the window
    d.loc[~d.fit, ["p_reco", "pz", "tandip", "d0", "rmax", "pdg", "fitcon", "chi2ndf", "nactive"]] = np.nan
    tab = RT.build(d)
    assert tab["time_eff"][0] == 0.0 and tab["time_eff"][-1] == 1.0
    v = _reference(10000, np.random.default_rng(11))
    t = np.random.default_rng(12).uniform(100, 1100, len(v))
    x, p = v[["e_x", "e_y", "e_z"]].to_numpy(), v[["e_px", "e_py", "e_pz"]].to_numpy()
    o = RD.apply(RD.to_device(tab), jax.random.PRNGKey(2), jax.numpy.asarray(v.has_entry.to_numpy()),
                 jax.numpy.asarray(v.nrow.to_numpy()), jax.numpy.asarray(x), jax.numpy.asarray(p), jax.numpy.asarray(t))
    fit = np.asarray(o["fit"])
    assert not fit[t < 200].any()
    late = t > 300
    on = RT.apply_numpy(tab, np.random.default_rng(3), v.has_entry.to_numpy(), v.nrow.to_numpy(), x, p, t=t)
    assert abs(fit[late].mean() - on["fit"][late].mean()) < 0.02


def _xp(v):
    return v[["e_x", "e_y", "e_z"]].to_numpy(), v[["e_px", "e_py", "e_pz"]].to_numpy()


def test_pools_are_opt_in_and_leave_the_other_tables_unchanged():
    d = _reference(20000, np.random.default_rng(20))
    a, b = RT.build(d), RT.build(d, pools=True)
    assert not any(k.startswith("pool_") for k in a)
    assert {k for k in b if k.startswith("pool_")} == {"pool_res", "pool_idx", "pool_start", "pool_count"}
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    assert b["pool_res"].shape[1] == len(RT.RES + RT.QUAL)
    assert (b["pool_count"] > 0).all() and (b["pool_count"] <= RT.POOL_MAX).all()


def test_pool_draw_is_one_reference_track_vector_of_the_bin():
    d = _reference(20000, np.random.default_rng(21))
    tab = RT.build(d, pools=True)
    v = _reference(3000, np.random.default_rng(22))
    x, p = _xp(v)
    o = RT.apply_numpy(tab, np.random.default_rng(23), v.has_entry.to_numpy(), v.nrow.to_numpy(), x, p)
    _, td, d0, rmax = RT.entry_truth(x, p, tab["axis_xy"], float(tab["bz"]))
    ne, nt = tab["eff_fit"].shape
    bn = np.clip(np.digitize(v.nrow.to_numpy(), tab["nrow_edges"]) - 1, 0, ne - 1)
    bt = np.clip(np.digitize(td, tab["tandip_edges"]) - 1, 0, nt - 1)
    q = o["qual"]
    vec = np.c_[o["p_reco"] - o["p_true"], o["tandip"] - td, o["d0"] - d0, o["rmax"] - rmax, o["fitcon"], o["chi2ndf"],
                o["nactive"]][q]
    for r, i, j in zip(vec[:300], bn[q][:300], bt[q][:300]):
        s, c = tab["pool_start"][i, j], tab["pool_count"][i, j]
        rows = tab["pool_res"][tab["pool_idx"][s:s + c]]
        assert np.isclose(rows, r, rtol=0, atol=1e-9).all(1).any()


def test_pool_draw_keeps_the_tail_dependence_the_copula_loses():
    """Momentum outliers that the stock fit gives an off-cut r_max: the CE-selected outlier fraction is ~0 in the
    reference and in the pool draw; the Gaussian copula lets many of them through."""
    def ref(n, seed):
        rng = np.random.default_rng(seed)
        d = _reference(n, rng)
        out = d.fit.to_numpy() & (rng.uniform(size=n) < 0.05)
        d.loc[out, "p_reco"] = d.p_true[out] + 2.0 + rng.exponential(0.5, out.sum())
        d.loc[out, "rmax"] = d.rmax_t[out] + 300.0                      # fails the r_max cut
        return d
    d = ref(60000, 24)
    v = ref(30000, 25)
    x, p = _xp(v)
    c = RT.CE_CUTS
    sel = v.fit & (v.fitcon > c["fitcon"]) & v.tandip.between(*c["tandip"]) & v.d0.between(*c["d0"]) & \
        v.rmax.between(*c["rmax"])
    stock = float((sel & (v.p_reco - v.p_true > 1.0)).sum() / sel.sum())
    frac = {}
    for pools in (False, True):
        o = RT.apply_numpy(RT.build(d, pools=pools), np.random.default_rng(26), v.has_entry.to_numpy(), v.nrow.to_numpy(),
                           x, p)
        frac[pools] = float((o["ce"] & (o["p_reco"] - o["p_true"] > 1.0)).sum() / o["ce"].sum())
    assert stock < 0.002 and frac[True] < 0.002
    assert frac[False] > 0.005


def test_device_pool_draw_matches_the_numpy_pool_draw():
    jax = pytest.importorskip("jax")
    from trkfastsim.recoresp import device as RD
    d = _reference(20000, np.random.default_rng(27))
    tab = RT.build(d, pools=True)
    v = _reference(20000, np.random.default_rng(28))
    x, p = _xp(v)
    on = RT.apply_numpy(tab, np.random.default_rng(29), v.has_entry.to_numpy(), v.nrow.to_numpy(), x, p)
    od = RD.apply(RD.to_device(tab), jax.random.PRNGKey(30), jax.numpy.asarray(v.has_entry.to_numpy()),
                  jax.numpy.asarray(v.nrow.to_numpy()), jax.numpy.asarray(x), jax.numpy.asarray(p))
    od = {k: np.asarray(a) for k, a in od.items()}
    for k in ("fit", "qual", "ce"):
        assert abs(od[k].mean() - on[k].mean()) < 0.012, k
    a, b = (o["p_reco"][o["qual"]] - o["p_true"][o["qual"]] for o in (od, on))
    assert abs(np.median(a) - np.median(b)) < 0.02
    assert abs(np.quantile(a, 0.05) - np.quantile(b, 0.05)) < 0.1
    vals = set(np.round(tab["pool_res"][:, 0], 9))
    assert all(round(float(r), 9) in vals for r in a[:200])         # device residuals are pool rows too


def test_pool_draw_leaves_the_t0_stream_unchanged():
    rng = np.random.default_rng(31)
    d = _reference(20000, rng)
    d["t_t"] = rng.uniform(500, 1500, len(d))
    d["t0"] = np.where(d.fit, d.t_t + rng.normal(1.2, 0.4, len(d)), np.nan)
    a, b = RT.build(d), RT.build(d, pools=True)
    v = _reference(3000, np.random.default_rng(32))
    x, p = _xp(v)
    t = np.full(len(v), 800.0)
    oa = RT.apply_numpy(a, np.random.default_rng(33), v.has_entry.to_numpy(), v.nrow.to_numpy(), x, p, t=t)
    ob = RT.apply_numpy(b, np.random.default_rng(33), v.has_entry.to_numpy(), v.nrow.to_numpy(), x, p, t=t)
    for k in ("fit", "qual"):
        assert np.array_equal(oa[k], ob[k])
    assert np.array_equal(oa["t0"], ob["t0"], equal_nan=True)


def test_shipped_pool_table_is_the_default_table_plus_pools():
    a, b = RT.load(), RT.load(RT.POOLS)
    assert set(b) - set(a) == {"pool_res", "pool_idx", "pool_start", "pool_count", "reference"}
    assert "CE" in str(b["reference"])
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    assert b["pool_res"].shape[1] == len(RT.RES + RT.QUAL) and (b["pool_count"] > 0).all()
    assert int(b["pool_idx"].max()) < len(b["pool_res"])


def test_reference_text_is_stored_only_when_given():
    d = _reference(5000, np.random.default_rng(34))
    assert "reference" not in RT.build(d)
    assert str(RT.build(d, reference="synthetic dev")["reference"]) == "synthetic dev"
