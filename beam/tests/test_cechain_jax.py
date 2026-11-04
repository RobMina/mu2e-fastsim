"""Unit tests: the device conversion-electron chain (beamfastsim.cechain_jax) against the NumPy chain (cechain)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from beamfastsim import cechain as C            # noqa: E402
from beamfastsim import cechain_jax as CJ       # noqa: E402
from beamfastsim import geometry as G           # noqa: E402

AXIS = -3904.0


def _geom(foil_z=(5800.0, 5822.0, 5844.0), ipa=(300.0, 0.5, 6900.0, 7900.0), opa=(454.0, 728.4, 20.0), ring=0.0):
    n = len(foil_z)
    beam = G.BeamGeometry(AXIS, np.array(foil_z), np.full(n, 21.5), np.full(n, 75.0), np.full(n, 0.0528),
                          4204.6, 25.4, 235.0, 525.0, ipa[0], ipa[1], ipa[2], ipa[3], opa[0], 4280.0, 8530.0)
    return C.CEGeometry(beam, opa[0], opa[1], opa[2], z_plane=8539.9, ring_t=ring, ring_hl=6.35 if ring else 0.0)


def _starts(n, rng, z=5700.0, r=40.0, cos=(0.5, 0.8)):
    ct = rng.uniform(*cos, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - ct ** 2)
    u = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    return np.tile([AXIS + r, 0.0, z], (n, 1)), u


VACUUM = dict(foil_z=(5000.0,), ipa=(300.0, 0.5, 9000.0, 9001.0), opa=(900.0, 900.0, 1.0))


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-9), (np.float32, 1e-4)])
def test_vacuum_tracks_and_recording_plane_equal_the_numpy_chain(dtype, tol):
    rng = np.random.default_rng(1)
    g = _geom(**VACUUM)
    x, u = _starts(300, rng, r=10.0, cos=(0.6, 0.8))
    B = [0.05, 0.0, 1.0]
    a = C.transport(rng, g, B, x, u, C.P_CE_AL, 3.0, z_record=7000.0)
    b = CJ.transport(jax.random.PRNGKey(0), g, B, x, u, C.P_CE_AL, 3.0, z_record=7000.0, dtype=dtype,
                     min_bucket=64)
    assert (a.fate == C.PLANE).all() and (b.fate == C.PLANE).all()
    for f in ("x", "u", "T", "t", "path", "rec_x", "rec_u", "rec_T", "rec_t"):
        va, vb = getattr(a, f), getattr(b, f)
        assert np.allclose(vb, va, rtol=tol, atol=tol if dtype == np.float64 else 0.2), f   # float32: 0.2 mm, ns
    assert b.rec_ok.all() and not (b.nfoil + b.nipa + b.nopa + b.nring).any()


def test_upstream_and_wall_losses_end_the_track():
    g = _geom(**VACUUM)
    x = np.tile([AXIS + 10.0, 0.0, 5700.0], (2, 1))
    u = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    res = CJ.transport(jax.random.PRNGKey(0), g, [0.0, 0.0, 0.0], x, u, C.P_CE_AL, 0.0, min_bucket=4)
    assert list(res.fate) == [C.UPSTREAM, C.WALL]
    assert res.path[0] < 5700.0 - g.z_up + 20.0 and res.path[1] < g.r_wall + 20.0


def test_foil_births_shells_and_rings_agree_with_the_numpy_chain():
    # electrons born in the foils, isotropic, 1 T axial: fates, crossings and the plane energy as the NumPy chain
    rng = np.random.default_rng(2)
    g = _geom(foil_z=tuple(5800.0 + 22.0 * np.arange(5)), ipa=(300.0, 0.5, 6000.0, 7000.0), ring=9.525)
    n = 1500
    k = rng.integers(0, 5, n)
    r = np.sqrt(rng.uniform(21.5 ** 2, 75.0 ** 2, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    x = np.stack([AXIS + r * np.cos(ph), r * np.sin(ph), g.beam.foil_z[k] + rng.uniform(-0.05, 0.05, n)], axis=1)
    u = C.isotropic(rng, n)
    a = C.transport(np.random.default_rng(3), g, [0.0, 0.0, 1.0], x, u, C.P_CE_AL, 0.0, backend="jax")
    b = CJ.transport(jax.random.PRNGKey(3), g, [0.0, 0.0, 1.0], x, u, C.P_CE_AL, 0.0, min_bucket=256)
    assert (b.nfoil >= 1).all() and (b.fate != C.LIVE).all()
    for fv in (C.PLANE, C.STOPPED, C.ABSORBED):
        fa, fb = (a.fate == fv).mean(), (b.fate == fv).mean()
        assert fb == pytest.approx(fa, abs=5 * np.sqrt(max(fa, 0.01) / n)), C.FATES[fv]
    for f in ("nfoil", "nipa", "nring"):
        assert getattr(b, f).mean() == pytest.approx(getattr(a, f).mean(), abs=0.1 + 0.1 * getattr(a, f).mean()), f
    pa, pb = a.fate == C.PLANE, b.fate == C.PLANE
    T0 = np.hypot(C.P_CE_AL, 0.511) - 0.511
    assert np.median(T0 - b.T[pb]) == pytest.approx(np.median(T0 - a.T[pa]), rel=0.15)
    assert np.allclose(b.x[pb, 2], g.z_plane, atol=1e-6)
    assert (b.t[pb] >= b.path[pb] / C.C_MM_NS - 1e-6).all()


def test_refill_and_unroll_keep_the_vacuum_result():
    # refilling 64 lanes from 300 starts, and the unrolled loops, give the same deterministic vacuum transport
    rng = np.random.default_rng(5)
    g = _geom(**VACUUM)
    x, u = _starts(300, rng, r=10.0, cos=(0.6, 0.8))
    a = CJ.transport(jax.random.PRNGKey(0), g, [0.0, 0.0, 1.0], x, u, C.P_CE_AL, 0.0, min_bucket=64)
    for kw in (dict(lanes=64, chunk=32), dict(unroll=True)):
        b = CJ.transport(jax.random.PRNGKey(0), g, [0.0, 0.0, 1.0], x, u, C.P_CE_AL, 0.0, min_bucket=64, **kw)
        assert np.array_equal(a.fate, b.fate)
        for f in ("x", "u", "t", "path", "nvac"):
            assert np.allclose(getattr(b, f), getattr(a, f), rtol=1e-10, atol=1e-9), f


def test_pair_comparison_of_two_runs_of_the_same_starts(tmp_path):
    # cecompare pair on two device runs of the same starts: same rows, paired yields and the start check
    from beamfastsim import cecompare as CC
    rng = np.random.default_rng(7)
    g = _geom(**VACUUM)
    x, u = _starts(200, rng, r=10.0, cos=(0.6, 0.8))
    for name, key in (("a", 0), ("b", 1)):
        r = CJ.transport(jax.random.PRNGKey(key), g, [0.0, 0.0, 1.0], x, u, C.P_CE_AL, 0.0, min_bucket=64)
        np.savez(tmp_path / f"{name}.npz", idx=np.arange(200), t0=np.zeros(200), wall=1.0,
                 **{k: v for k, v in r._asdict().items() if v is not None})
    out = CC.pair(str(tmp_path / "a.npz"), str(tmp_path / "b.npz"), g, lambda xx: np.tile([0.0, 0.0, 1.0], (len(xx), 1)))
    c = out["cuts"]["450.0"]
    assert out["rows"] == 200 and c["yield_per_start"]["diff"] == 0.0
    assert c["same_start"]["dp"]["n"] == 0 or abs(c["same_start"]["dp"]["mean"]) < 1e-9
    np.savez(tmp_path / "c.npz", idx=np.arange(200)[::-1], t0=np.zeros(200), wall=1.0,
             **{k: v for k, v in r._asdict().items() if v is not None})
    with pytest.raises(ValueError, match="same starts"):
        CC.pair(str(tmp_path / "a.npz"), str(tmp_path / "c.npz"), g, lambda xx: np.zeros((len(xx), 3)))
