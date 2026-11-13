"""sim.fieldsteps_jax: the device twin of the field-on electron transport (synthetic geometry, no data files)."""
import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")

from mu2efastsim import materials  # noqa: E402
from trkfastsim import geometry  # noqa: E402
from trkfastsim.sim import fieldsteps as fs  # noqa: E402
from trkfastsim.sim import fieldsteps_jax as FJ  # noqa: E402

NSTRAW = 48


def _geometry():
    """Two layers of x-axis straws (y pitch 6 mm, layers at z = -2.5 and +2.5), 900 mm long, about the origin."""
    n = NSTRAW
    y = (np.arange(n) // 2) * 6.0 - 70.0
    z = np.where(np.arange(n) % 2 == 0, -2.5, 2.5)
    zeros = np.zeros(n, int)
    return geometry.StrawGeometry(
        plane=zeros, panel=zeros, straw=np.arange(n), mid=np.c_[np.zeros(n), y, z], dir=np.tile([1.0, 0, 0], (n, 1)),
        halflen=np.full(n, 450.0), radius=2.5085, panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=2.5235, metal_t=0.0001, wire_r=0.0125,
        wire_plate_t=0.0002)


def _grid(B):
    """A uniform field B [T] on a regular grid covering the test volume (G4BL layout: lo, d, n, B[nx, ny, nz, 3])."""
    n = np.array([5, 5, 5])
    return np.array([-1000.0, -1000.0, -1000.0]), np.array([500.0, 500.0, 500.0]), n, np.broadcast_to(
        np.asarray(B, float), (5, 5, 5, 3)).copy()


def _start(o, d, p):
    n = len(o)
    return pd.DataFrame({"run": 1, "subrun": 1, "event": np.arange(n), "sp": 1, "pdg": 11,
                         "x": o[:, 0], "y": o[:, 1], "z": o[:, 2], "px": d[:, 0] * p, "py": d[:, 1] * p,
                         "pz": d[:, 2] * p, "t": 0.0})


def _lines(n, rng):
    o = np.c_[rng.uniform(-300, 300, n), rng.uniform(-60, 60, n), np.full(n, -40.0)]
    d = np.c_[rng.normal(0, 0.05, n), rng.normal(0, 0.3, n), np.ones(n)]
    return o, d / np.linalg.norm(d, axis=1)[:, None]


def _setup(B, **kw):
    geom = _geometry()
    env = fs.Envelope(geom, pad_r=10.0, pad_z=10.0, axis_xy=(0.0, 0.0))
    trk = FJ.Tracker(geom, _grid(B), None, env, np.zeros(3), np.asarray(B, float), **kw)
    return geom, env, trk


def _no_loss(rng, m, p, ln, ms):
    return np.zeros(len(p))


def test_without_loss_and_scattering_the_device_rows_equal_numpy():
    B = np.array([0.0, 0.0, 1.0])
    geom, env, trk = _setup(B, chord_mm=5.0, eloss_on=False, msc=False)
    o, d = _lines(200, np.random.default_rng(3))
    start = _start(o, d, 105.0)
    ref, rfin = fs.transport(np.random.default_rng(4), start, geom, lambda x: np.tile(B, (len(x), 1)), chord_mm=5.0,
                             envelope=env, origin=np.zeros(3), bnom=B, msc=False, loss=_no_loss, tail=False)
    res = FJ.transport(trk, jax.random.PRNGKey(1), o, d * 105.0, np.zeros(len(o)))
    dev = FJ.to_frame(res, start)
    assert len(dev) == len(ref) > 100
    a = dev.sort_values(["event", "t"]).reset_index(drop=True)
    b = ref.sort_values(["event", "t"]).reset_index(drop=True)
    assert np.array_equal(a.straw.to_numpy(), b.straw.to_numpy())
    assert np.array_equal(a.nsgs.to_numpy(), b.nsgs.to_numpy())
    for c in ("len", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz", "width"):
        np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy(), atol=1e-8)
    fin = FJ.final_frame(res, start)
    np.testing.assert_allclose(fin[["x", "y", "z"]].to_numpy(), rfin[["x", "y", "z"]].to_numpy(), atol=1e-8)
    assert np.array_equal(fin.status.to_numpy(), rfin.status.to_numpy())
    assert all(v == 0 for k, v in res.stats.items() if k.endswith("overflow"))


@pytest.mark.slow
def test_electron_loss_pairs_follow_the_numpy_model():
    """Mean and spread of the device per-(track, material) loss against fieldsteps.electron_loss."""
    _, _, trk = _setup(np.array([0.0, 0.0, 1.0]))
    n = 40000
    for mi, ln in ((len(trk.mats) - 3, 5.0), (len(trk.mats) - 5, 0.015), (len(trk.mats) - 2, 0.02)):
        m = trk.mats[mi]
        ref = fs.electron_loss(np.random.default_rng(5), m, np.full(n, 104.0), np.full(n, ln))
        dev, dcap, bcap = FJ._electron_loss(jax.random.PRNGKey(6), trk.T, trk.mc, trk.lnt0, trk.dlnt,
                                            jax.numpy.full(n, 104.0), jax.numpy.full(n, ln),
                                            jax.numpy.full(n, mi, jax.numpy.int32))
        dev = np.asarray(dev)
        se = np.hypot(ref.std(), dev.std()) / np.sqrt(n)
        assert abs(dev.mean() - ref.mean()) < 5 * se + 1e-6, m.name
        assert int(dcap) == 0 and int(bcap) == 0
        np.testing.assert_allclose(np.median(dev), np.median(ref), rtol=0.05)


@pytest.mark.slow
def test_full_physics_transport_agrees_with_numpy_statistically_and_is_seeded():
    B = np.array([0.0, 0.0, 1.0])
    geom, env, trk = _setup(B)
    o, d = _lines(400, np.random.default_rng(7))
    start = _start(o, d, 105.0)
    ref, rfin = fs.transport(np.random.default_rng(8), start, geom, lambda x: np.tile(B, (len(x), 1)), envelope=env,
                             origin=np.zeros(3), bnom=B, loss=fs.electron_loss, tail=False)
    r1 = FJ.transport(trk, jax.random.PRNGKey(9), o, d * 105.0, np.zeros(len(o)))
    r2 = FJ.transport(trk, jax.random.PRNGKey(9), o, d * 105.0, np.zeros(len(o)))
    for k in r1.steps:
        assert np.array_equal(np.asarray(r1.steps[k]), np.asarray(r2.steps[k]), equal_nan=True)
    dev = FJ.to_frame(r1, start)
    assert abs(len(dev) - len(ref)) < 5 * np.sqrt(len(ref)) * 0.2 + 10
    for c in ("len", "eion"):
        a, b = dev[c].to_numpy(), ref[c].to_numpy()
        se = np.hypot(a.std() / np.sqrt(len(a)), b.std() / np.sqrt(len(b)))
        assert abs(a.mean() - b.mean()) < 5 * se, c
    fin = FJ.final_frame(r1, start)
    pd_, pr = np.linalg.norm(fin[["px", "py", "pz"]].to_numpy(), axis=1), np.linalg.norm(rfin[["px", "py", "pz"]].to_numpy(), axis=1)
    assert abs(np.median(pd_) - np.median(pr)) < 0.05


def test_inactive_tracks_get_no_steps():
    B = np.array([0.0, 0.0, 1.0])
    _, _, trk = _setup(B)
    o, d = _lines(50, np.random.default_rng(10))
    act = np.arange(50) % 2 == 0
    res = FJ.transport(trk, jax.random.PRNGKey(11), o, d * 105.0, np.zeros(50), active=act)
    v = np.asarray(res.steps["valid"])
    assert not v[~act].any() and v[act].any()
    assert (np.asarray(res.final["status"])[~act] == FJ.LEFT).all()


def test_thin_materials_are_the_numpy_set():
    _, _, trk = _setup(np.array([0.0, 0.0, 1.0]))
    names = [m.name for m in trk.mats[-5:]]
    assert names == [materials.MYLAR.name, materials.ALUMINIUM_6061.name, trk.gas.name, materials.TUNGSTEN.name,
                     materials.GOLD.name]


def test_a_chord_budget_caps_the_slow_loopers_only():
    """max_chords: tracks still live after the budget end CAPPED with their records; fast tracks that finished within
    it are unchanged. No loss and no scattering, so the loopers drift a fixed ~1 mm per turn and need ~1,000 chords."""
    B = np.array([0.0, 0.0, 1.0])
    _, _, trk = _setup(B, eloss_on=False, msc=False)
    o, d = _lines(20, np.random.default_rng(12))
    lo = np.c_[np.zeros(4), np.linspace(-20, 20, 4), np.zeros(4)]              # 5 MeV/c loopers, almost transverse
    ld = np.tile([1.0, 0.0, 0.01], (4, 1)) / np.linalg.norm([1.0, 0.0, 0.01])
    x, u = np.r_[o, lo], np.r_[d, ld]
    p = np.r_[np.full(20, 105.0), np.full(4, 5.0)][:, None] * u
    a = FJ.transport(trk, jax.random.PRNGKey(13), x, p, np.zeros(24))
    b = FJ.transport(trk, jax.random.PRNGKey(13), x, p, np.zeros(24), max_chords=256)
    sa, sb = np.asarray(a.final["status"]), np.asarray(b.final["status"])
    assert (sb[20:] == FJ.CAPPED).all() and (sa[20:] == FJ.LEFT).all()
    assert b.stats["iterations"] == 256 and a.stats["iterations"] > 256
    for k in ("valid", "eion", "x0", "t"):
        assert np.array_equal(np.asarray(a.steps[k])[:20], np.asarray(b.steps[k])[:20]), k
    va, vb = np.asarray(a.steps["valid"])[20:], np.asarray(b.steps["valid"])[20:]
    assert vb.any() and (vb.sum(1) <= va.sum(1)).all() and (vb.sum() < va.sum())



@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_positron_rows_are_the_electron_rows_in_the_reversed_field(dtype):
    # without loss and scattering a positron in B crosses the straws of an electron in -B, record for record
    B = np.array([0.0, 0.2, 1.0])
    _, _, tp = _setup(B, chord_mm=5.0, eloss_on=False, msc=False, dtype=dtype)
    _, _, te = _setup(-B, chord_mm=5.0, eloss_on=False, msc=False, dtype=dtype)
    o, d = _lines(200, np.random.default_rng(21))
    e = FJ.transport(te, jax.random.PRNGKey(1), o, d * 60.0, np.zeros(len(o)))
    p = FJ.transport(tp, jax.random.PRNGKey(1), o, d * 60.0, np.zeros(len(o)), positron=True)
    ve, vp = np.asarray(e.steps["valid"]), np.asarray(p.steps["valid"])
    assert np.array_equal(ve, vp) and ve.sum() > 100
    tol = 1e-8 if dtype == np.float64 else 2e-3
    assert np.array_equal(np.asarray(e.steps["straw"])[ve], np.asarray(p.steps["straw"])[vp])
    for c in ("len", "t", "x0", "y1", "z1", "px", "pz"):
        np.testing.assert_allclose(np.asarray(p.steps[c])[vp], np.asarray(e.steps[c])[ve], atol=tol, rtol=tol)
    # the same positron in -B (the electron's field) bends the other way
    q = FJ.transport(te, jax.random.PRNGKey(1), o, d * 60.0, np.zeros(len(o)), positron=True)
    fe, fq = np.asarray(e.final["x"]), np.asarray(q.final["x"])
    assert np.abs(fe - fq).max() > 1.0


def test_positron_loss_tables_are_bhabha_and_the_full_transport_runs():
    B = np.array([0.0, 0.0, 1.0])
    _, _, trk = _setup(B)
    from mu2efastsim import eslab_jax as SJ
    dedx_e, dedx_p = np.asarray(trk.T["dedx"]), np.asarray(trk.T["p_dedx"])
    assert dedx_e.shape == dedx_p.shape and not np.allclose(dedx_e, dedx_p)
    hi = SJ._grid()[1] > 1.0                                        # above 1 MeV: e+ and e- loss within ~10 %
    np.testing.assert_allclose(dedx_p[:, hi], dedx_e[:, hi], rtol=0.15)
    assert not np.allclose(np.asarray(trk.T["gas_dedx"]), np.asarray(trk.T["p_gas_dedx"]))
    o, d = _lines(400, np.random.default_rng(22))
    e = FJ.transport(trk, jax.random.PRNGKey(5), o, d * 105.0, np.zeros(len(o)))
    p = FJ.transport(trk, jax.random.PRNGKey(5), o, d * 105.0, np.zeros(len(o)), positron=True)
    p2 = FJ.transport(trk, jax.random.PRNGKey(5), o, d * 105.0, np.zeros(len(o)), positron=True)
    for k in p.steps:                                               # seeded
        assert np.array_equal(np.asarray(p.steps[k]), np.asarray(p2.steps[k]), equal_nan=True)
    ne, np_ = int(np.asarray(e.steps["valid"]).sum()), int(np.asarray(p.steps["valid"]).sum())
    assert abs(np_ - ne) < 0.1 * ne
    ee = np.asarray(e.steps["eion"])[np.asarray(e.steps["valid"])]
    ep = np.asarray(p.steps["eion"])[np.asarray(p.steps["valid"])]
    assert abs(ep.mean() - ee.mean()) < 0.1 * ee.mean()
