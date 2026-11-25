"""Device chain (JAX) against the host reference on synthetic banks: same physics, statistical agreement."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from beamfastsim import fieldmap  # noqa: E402
from beamfastsim import ptgt_chain as C  # noqa: E402
from beamfastsim import ptgt_chain_jax as J  # noqa: E402
from beamfastsim import ptgt_device as D  # noqa: E402

MATS = ("G4_W",)
EDGES = np.linspace(0.0, 4.0, 9)


def _uniform_map(bz):
    lo = np.array([-2000.0, -2000.0, -3000.0])
    n = np.array([3, 3, 3])
    B = np.zeros((3, 3, 3, 3)); B[..., 2] = bz
    return fieldmap.FieldMap(lo, np.array([2000.0, 2000.0, 3000.0]), n, B)


def _bank(tmp_path, n_pim=1, sigma_p=1e3, dedx=0.0, edep_local=0.0):
    shape = (len(D.SPECIES), len(MATS), EDGES.size - 1)
    cells = {k: np.zeros(shape) for k in D.CELL_ARRAYS}
    cells["x0"][:] = 1e12
    cells["sigma_inel"][0] = sigma_p
    cells["dedx"][:] = dedx
    cells["el_cos_icdf"] = np.ones(shape + (D.N_EL,))
    ncell = int(np.prod(shape))
    per = np.array([n_pim if c // (len(MATS) * (EDGES.size - 1)) == 0 else 0 for c in range(ncell)])
    trials = dict(trial_cell=np.arange(ncell), sec_offsets=np.concatenate([[0], np.cumsum(per)]),
                  edep_local=np.full(ncell, float(edep_local)))
    ns = int(per.sum())
    rng = np.random.default_rng(0)
    d = rng.normal(size=(ns, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    secs = dict(sec_pdg=np.full(ns, -211, np.int32), sec_p=d * 150.0)
    fields = dict(name="beam.thin_target.synthetic", source=dict(samples=[], description="synthetic"),
                  generator=dict(name="test", release="none", physics_list="none", configuration="none"),
                  release=dict(repository="test", commit="none", entry_point="test"), seeds=dict(build=1),
                  config_hash=None, split=dict(training="all", held_out="none", rule="synthetic"),
                  unrecorded=dict(config_hash="synthetic"))
    D.save_bank(tmp_path / "b.npz", MATS, EDGES, cells, trials, secs, fields)
    return D.Bank.load(tmp_path / "b.npz")


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_device_matches_host_on_synthetic_cascade(tmp_path, dtype):
    bank = _bank(tmp_path, n_pim=3)
    tgt = D.Target.rod(100.0, 10.0, 10.0, 10.0)
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0)
    fm = _uniform_map(2.0)
    rng = np.random.default_rng(3)
    n = 4000
    prim = C.primaries(n, tgt, rng)
    host = C.run(bank, tgt, lambda x: np.tile([0.0, 0.0, 2.0], (len(x), 1)), env, prim, np.random.default_rng(4))
    rec, cnt = J.run(bank, tgt, fm, env, prim, seed=5, pool=4096, steps=32, dtype=dtype)
    assert cnt["inelastic"] == host["counts"]["inelastic"] == n          # every proton interacts once
    nh, nd = host["counts"]["recorded"], len(rec["pdg"])
    assert nh > 2000, (nh, nd)
    assert abs(nd - nh) < 4 * np.sqrt(nh), (nh, nd)                        # same physics, independent streams
    assert set(np.unique(rec["pdg"])) <= {13, -13, 211, -211}
    assert np.allclose(rec["x"][:, 2], 800.0, atol=1e-2)
    assert cnt["step_cap"] == 0
    ph, pd = np.linalg.norm(host["record"]["p"], axis=1), np.linalg.norm(rec["p"], axis=1)
    assert abs(np.median(ph) - np.median(pd)) < 0.1 * np.median(ph)


def test_refill_moves_emissions_into_dead_lanes():
    N, E = 8, 16
    L = J.Lanes(alive=jnp.array([True, False] * 4), sp=jnp.zeros(N, jnp.int32), x=jnp.zeros((N, 3)), u=jnp.zeros((N, 3)),
                p=jnp.ones(N), t=jnp.zeros(N), tau=jnp.full(N, jnp.inf), seg=jnp.full(N, -1, jnp.int32), ev=jnp.zeros(N, jnp.int64),
                n=jnp.zeros(N, jnp.int32), pk=jnp.zeros(N, jnp.int32), pj=jnp.zeros(N, jnp.int64), pch=jnp.zeros(N, jnp.int32))
    emit = J._empty_emit(E, jnp.float64)
    emit = (emit[0], emit[1].at[:6].set(J.species_of(np.array([211] * 6))), emit[2], emit[3], emit[4].at[:6].set(7.0),
            emit[5], emit[6], emit[7].at[:6].set(jnp.arange(6)))
    L2, emit2, left = J._refill(L, emit, jnp.int64(6), jax.random.PRNGKey(0), 1.0)
    assert int(jnp.sum(L2.alive)) == 8 and int(left) == 2
    assert np.allclose(np.asarray(L2.p)[1::2], 7.0) and list(np.asarray(emit2[7][:2])) == [4, 5]
    assert np.all(np.isfinite(np.asarray(L2.tau)[1::2]))                   # pions get a lifetime


def test_float32_target_far_from_origin_finishes(tmp_path):
    """Boundary steps keep moving in float32 when the target sits at Mu2e coordinates (the frame is shifted)."""
    bank = _bank(tmp_path, n_pim=2)
    tgt = D.Target.rod(100.0, 10.0, 10.0, 10.0, origin=np.array([3904.0, 0.0, -6164.5]))
    env = C.Envelope(z_nodes=np.array([-9000.0, -3000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-8000.0,
                     z_plane=-5300.0, r_plane=1800.0, axis_x=3904.0)
    lo = np.array([3904.0 - 2000.0, -2000.0, -9000.0])
    B = np.zeros((3, 3, 3, 3)); B[..., 2] = 2.0
    fm = fieldmap.FieldMap(lo, np.array([2000.0, 2000.0, 3000.0]), np.array([3, 3, 3]), B)
    prim = C.primaries(1000, tgt, np.random.default_rng(8))
    rec, cnt = J.run(bank, tgt, fm, env, prim, seed=9, pool=2048, steps=32, dtype=jnp.float32, max_chunks=20000)
    assert cnt["inelastic"] == 1000 and cnt["step_cap"] == 0
    assert len(rec["pdg"]) > 100 and np.allclose(rec["x"][:, 2], -5300.0, atol=0.05)


def _bank_two_rows(tmp_path, sigma_p, rows=(1, 9)):
    """Proton cells with one trial per multiplicity in `rows`; secondaries are 150 MeV/c protons (below the nucleon
    cut: each is counted once as stopped). No other species interacts."""
    shape = (len(D.SPECIES), len(MATS), EDGES.size - 1)
    cells = {k: np.zeros(shape) for k in D.CELL_ARRAYS}
    cells["x0"][:] = 1e12
    cells["sigma_inel"][0] = sigma_p
    cells["el_cos_icdf"] = np.ones(shape + (D.N_EL,))
    nb = EDGES.size - 1
    tc = np.repeat(np.arange(nb), len(rows))                       # proton cells are the first nb flat cells
    per = np.tile(np.asarray(rows), nb)
    trials = dict(trial_cell=tc, sec_offsets=np.concatenate([[0], np.cumsum(per)]), edep_local=np.zeros(tc.size))
    ns = int(per.sum())
    secs = dict(sec_pdg=np.full(ns, 2212, np.int32), sec_p=np.tile([[0.0, 0.0, 150.0]], (ns, 1)))
    fields = dict(name="beam.thin_target.synthetic", source=dict(samples=[], description="synthetic"),
                  generator=dict(name="test", release="none", physics_list="none", configuration="none"),
                  release=dict(repository="test", commit="none", entry_point="test"), seeds=dict(build=1),
                  config_hash=None, split=dict(training="all", held_out="none", rule="synthetic"),
                  unrecorded=dict(config_hash="synthetic"))
    D.save_bank(tmp_path / "b2.npz", MATS, EDGES, cells, trials, secs, fields)
    return D.Bank.load(tmp_path / "b2.npz")


@pytest.mark.parametrize("pool,emit_cap", [(4096, None), (32, 64)])
def test_deferral_keeps_the_sampled_outcome(tmp_path, pool, emit_cap):
    """Interaction probability 1/2 in the rod and rows of 1 or 9 secondaries. A tiny emission buffer defers most
    emitters; the interaction fraction and the mean multiplicity must not move (redrawing a deferred step would
    lower both)."""
    L_rod = 100.0
    bank = _bank_two_rows(tmp_path, sigma_p=np.log(2.0) / L_rod)
    tgt = D.Target.rod(L_rod, 10.0, 10.0, 10.0)
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-300.0,
                     z_plane=300.0, r_plane=1800.0, axis_x=0.0)
    n = 8000
    prim = C.primaries(n, tgt, np.random.default_rng(11))
    rec, cnt = J.run(bank, tgt, _uniform_map(0.0), env, prim, seed=12, pool=pool, emit_cap=emit_cap, steps=64,
                     dtype=jnp.float64, max_chunks=200000)
    if emit_cap:
        assert cnt["deferred"] > n // 4, cnt                       # the test exercises deferral
    frac = cnt["inelastic"] / n
    assert abs(frac - 0.5) < 4 * np.sqrt(0.25 / n), cnt
    mult = cnt["stopped"] / cnt["inelastic"]
    assert abs(mult - 5.0) < 4 * 4.0 / np.sqrt(cnt["inelastic"]), cnt


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_device_matches_host_with_slab(tmp_path, dtype):
    """Synthetic cascade with a full-disk slab before the plane (absorption 1 - exp(-0.4) at normal incidence, loss
    and scattering): device and host agree within statistics on recorded lanes, slab absorption and momentum."""
    bank = _bank(tmp_path, n_pim=3)
    shape = (len(D.SPECIES), len(MATS), EDGES.size - 1)
    cells = {k: np.zeros(shape) for k in D.CELL_ARRAYS}
    cells["sigma_inel"][:] = 0.02; cells["dedx"][:] = 0.5; cells["x0"][:] = 20.0
    cells["el_cos_icdf"] = np.ones(shape + (D.N_EL,))
    fields = dict(name="beam.thin_target.synthetic", source=dict(samples=[], description="synthetic"),
                  generator=dict(name="test", release="none", physics_list="none", configuration="none"),
                  release=dict(repository="test", commit="none", entry_point="test"), seeds=dict(build=1),
                  config_hash=None, split=dict(training="all", held_out="none", rule="synthetic"),
                  unrecorded=dict(config_hash="synthetic"))
    D.save_bank(tmp_path / "slab.npz", MATS, EDGES, cells,
                dict(trial_cell=np.zeros(0, np.int64), sec_offsets=np.zeros(1, np.int64), edep_local=np.zeros(0)),
                dict(sec_pdg=np.zeros(0, np.int32), sec_p=np.zeros((0, 3))), fields)
    slab = (D.Bank.load(tmp_path / "slab.npz"), 0)
    tgt = D.Target.rod(100.0, 10.0, 10.0, 10.0)
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0, slabs=((400.0, 0.0, 1850.0, 20.0),))
    prim = C.primaries(4000, tgt, np.random.default_rng(3))
    host = C.run(bank, tgt, lambda x: np.tile([0.0, 0.0, 2.0], (len(x), 1)), env, prim, np.random.default_rng(4),
                 slab=slab)
    rec, cnt = J.run(bank, tgt, _uniform_map(2.0), env, prim, seed=5, pool=4096, steps=32, dtype=dtype, slab=slab)
    hc = host["counts"]
    nh, nd = hc["recorded"], len(rec["pdg"])
    assert hc["slab_absorbed"] > 500 and nh > 1000, hc
    assert abs(nd - nh) < 4 * np.sqrt(nh), (nh, nd)
    assert abs(cnt["slab_absorbed"] - hc["slab_absorbed"]) < 5 * np.sqrt(hc["slab_absorbed"]), (cnt, hc)
    ph, pd = np.linalg.norm(host["record"]["p"], axis=1), np.linalg.norm(rec["p"], axis=1)
    assert abs(np.median(ph) - np.median(pd)) < 0.05 * np.median(ph)
    assert np.allclose(rec["x"][:, 2], 800.0, atol=1e-2)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_device_locate_and_cascade_on_a_clad_rod(tmp_path, dtype):
    bank = _bank(tmp_path, n_pim=3)
    tgt = D.Target.clad_rod(100.0, 6.0, 4.0, mat_core=0, mat_clad=0)       # two segments over one z range
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0)
    fm = _uniform_map(2.0)
    S, nk, nmat = J.make_static(bank, tgt, env, fm, dtype)
    cfg = J.make_cfg(tgt, env, C.Stepping(), nk, nmat, dtype)
    rng = np.random.default_rng(5)
    xl = np.stack([rng.uniform(-12, 12, 4000), rng.uniform(-12, 12, 4000), rng.uniform(-60, 60, 4000)], 1)
    r = np.hypot(xl[:, 0], xl[:, 1])
    far = (np.abs(r - 6.0) > 1e-3) & (np.abs(r - 10.0) > 1e-3) & (np.abs(np.abs(xl[:, 2]) - 50.0) > 1e-3)
    dev = np.asarray(J.locate(S, cfg, jnp.asarray(xl, dtype)))
    assert np.array_equal(dev[far], tgt.locate(xl)[far]) and {0, 1, -1} <= set(dev.tolist())
    prim = C.primaries(3000, tgt, np.random.default_rng(3))
    host = C.run(bank, tgt, lambda x: np.tile([0.0, 0.0, 2.0], (len(x), 1)), env, prim, np.random.default_rng(4))
    rec, cnt = J.run(bank, tgt, fm, env, prim, seed=5, pool=4096, steps=32, dtype=dtype)
    assert cnt["inelastic"] == host["counts"]["inelastic"] == 3000
    nh, nd = host["counts"]["recorded"], len(rec["pdg"])
    assert nh > 1500 and abs(nd - nh) < 4 * np.sqrt(nh), (nh, nd)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_device_spheres_cylinders_match_host(tmp_path, dtype):
    """The RAL target (rod, hubs, hub balls, spokes): device locate / distances against the host."""
    from beamfastsim import ptgt_deck as K
    bank = _bank(tmp_path, n_pim=3)
    tgt = K.ral_target(0)
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0)
    S, nk, nmat = J.make_static(bank, tgt, env, _uniform_map(2.0), dtype)
    cfg = J.make_cfg(tgt, env, C.Stepping(), nk, nmat, dtype)
    assert cfg.ncone == 5 and cfg.nseg == 17
    rng = np.random.default_rng(8)
    n = 15000
    xl = np.concatenate([np.stack([rng.uniform(-18, 18, n), rng.uniform(-18, 18, n), rng.uniform(-100, 100, n)], 1),
                         np.stack([rng.uniform(-150, 150, n), rng.uniform(-150, 150, n), rng.uniform(-120, 120, n)], 1)])
    # points near the spokes: along each axis, jittered by about its radius
    c = tgt.cy
    f = rng.uniform(0, 1, (n, 1)); k = rng.integers(0, len(c), n)
    xl = np.concatenate([xl, c[k, 0:3] + f * (c[k, 3:6] - c[k, 0:3]) + rng.normal(scale=0.4, size=(n, 3)),
                         tgt.sp[k, 0:3] + rng.normal(scale=1.5, size=(n, 3))])          # and around the balls
    host = tgt.locate(xl)
    dev = np.asarray(J.locate(S, cfg, jnp.asarray(xl, dtype)))
    same = dev == host
    assert np.mean(same) > 0.9995 and set(host.tolist()) == set(range(-1, 17))
    assert np.array_equal(np.asarray(J.cylinder_near(S, jnp.asarray(xl, dtype), 1.0)), tgt.cylinder_near(xl, 1.0))
    dl = rng.normal(size=xl.shape); dl /= np.linalg.norm(dl, axis=1, keepdims=True)
    x, d = jnp.asarray(xl, dtype), jnp.asarray(dl, dtype)
    inside = same & (host >= 0)
    do = np.asarray(J.distance_out(S, cfg, x, d, jnp.asarray(host, jnp.int32)))[inside]
    assert np.mean(np.isclose(do, tgt.distance_out(xl[inside], dl[inside], host[inside]), rtol=1e-4, atol=1e-3)) > 0.999
    out = same & (host < 0)
    di, dh = np.asarray(J.distance_in(S, cfg, x, d))[out], tgt.distance_in(xl[out], dl[out])
    fin = np.isfinite(dh)
    assert np.mean(np.isfinite(di) == fin) > 0.999 and fin.sum() > 2000
    both = fin & np.isfinite(di)
    assert np.mean(np.isclose(di[both], dh[both], rtol=1e-4, atol=1e-3)) > 0.999
    prim = C.primaries(200, tgt, np.random.default_rng(3))
    assert np.allclose(tgt.to_local(prim["x"])[:, 2], 80.5)     # the deck's beam start, inside the upstream hub


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_device_cascade_on_the_ral_target(tmp_path, dtype):
    from beamfastsim import ptgt_deck as K
    bank = _bank(tmp_path, n_pim=3)
    tgt = K.ral_target(0)
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0)
    prim = C.primaries(300, tgt, np.random.default_rng(3))
    rec, cnt = J.run(bank, tgt, _uniform_map(2.0), env, prim, seed=5, pool=4096, steps=32, dtype=dtype)
    assert cnt["inelastic"] > 150


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_device_many_segments_match_host(tmp_path, dtype):
    """A Hayman-like body: disjoint discs, a frustum, a hollow end ring and fins (segment axis vectorized)."""
    bank = _bank(tmp_path, n_pim=3)
    zr = np.array([[-60.0, -50.0], [-48.0, -30.0], [-28.0, -10.0], [-8.0, 10.0], [12.0, 30.0], [32.0, 50.0], [50.0, 60.0]])
    r_in = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [4.0, 4.0]])
    r_out = np.array([[8, 8], [6, 6], [6, 6], [5, 7], [6, 6], [6, 6], [8.0, 8.0]])
    boxes = np.array([[-1.0, 1.0, 6.0, 12.0, -48.0, -30.0], [6.0, 12.0, -1.0, 1.0, 12.0, 30.0]])
    tgt = D.Target(np.array([-60.0, 60.0]), r_in, r_out, np.zeros(9, int), z_ranges=zr, boxes=boxes)
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0)
    S, nk, nmat = J.make_static(bank, tgt, env, _uniform_map(2.0), dtype)
    cfg = J.make_cfg(tgt, env, C.Stepping(), nk, nmat, dtype)
    assert cfg.ncone == 7 and cfg.nseg == 9
    rng = np.random.default_rng(7)
    xl = np.stack([rng.uniform(-14, 14, 20000), rng.uniform(-14, 14, 20000), rng.uniform(-65, 65, 20000)], 1)
    host = tgt.locate(xl)
    dev = np.asarray(J.locate(S, cfg, jnp.asarray(xl, dtype)))
    assert np.mean(dev == host) > 0.9995 and set(host.tolist()) == set(range(-1, 9))
    same = dev == host
    dl = rng.normal(size=xl.shape); dl /= np.linalg.norm(dl, axis=1, keepdims=True)
    x, d = jnp.asarray(xl, dtype), jnp.asarray(dl, dtype)
    inside = same & (host >= 0)
    do = np.asarray(J.distance_out(S, cfg, x, d, jnp.asarray(host, jnp.int32)))[inside]
    assert np.mean(np.isclose(do, tgt.distance_out(xl[inside], dl[inside], host[inside]), rtol=1e-4, atol=1e-3)) > 0.999
    out = same & (host < 0)
    di, dh = np.asarray(J.distance_in(S, cfg, x, d))[out], tgt.distance_in(xl[out], dl[out])
    fin = np.isfinite(dh)
    assert np.mean(np.isfinite(di) == fin) > 0.999 and np.isfinite(dh).sum() > 1000
    both = fin & np.isfinite(di)
    assert np.mean(np.isclose(di[both], dh[both], rtol=1e-4, atol=1e-3)) > 0.999


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_device_boxes_match_host(tmp_path, dtype):
    bank = _bank(tmp_path, n_pim=3)
    tgt = D.Target(np.array([-50.0, 50.0]), np.zeros((1, 2)), np.full((1, 2), 6.0), np.array([0, 0, 0, 0]),
                   boxes=np.array([[-1.0, 1.0, 6.0, 20.0, -50.0, 50.0], [6.0, 20.0, -1.0, 1.0, -50.0, 50.0],
                                   [-1.0, 1.0, -20.0, -6.0, -40.0, 40.0]]))
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0)
    fm = _uniform_map(2.0)
    S, nk, nmat = J.make_static(bank, tgt, env, fm, dtype)
    cfg = J.make_cfg(tgt, env, C.Stepping(), nk, nmat, dtype)
    assert cfg.ncone == 1 and cfg.nseg == 4
    rng = np.random.default_rng(6)
    xl = np.stack([rng.uniform(-25, 25, 6000), rng.uniform(-25, 25, 6000), rng.uniform(-60, 60, 6000)], 1)
    host = tgt.locate(xl)
    edges = np.array([-50, -40, -20, -6, -1, 1, 6, 20, 40, 50.0])
    far = np.all(np.abs(xl[:, :, None] - edges[None, None, :]) > 1e-3, axis=(1, 2))
    dev = np.asarray(J.locate(S, cfg, jnp.asarray(xl, dtype)))
    assert np.array_equal(dev[far], host[far]) and set(host.tolist()) == {-1, 0, 1, 2, 3}
    dl = rng.normal(size=xl.shape); dl /= np.linalg.norm(dl, axis=1, keepdims=True)
    inside = far & (host >= 0)
    do = np.asarray(J.distance_out(S, cfg, jnp.asarray(xl, dtype), jnp.asarray(dl, dtype), jnp.asarray(host, jnp.int32)))
    assert np.allclose(do[inside], tgt.distance_out(xl[inside], dl[inside], host[inside]), rtol=1e-4, atol=1e-3)
    out = far & (host < 0)
    di = np.asarray(J.distance_in(S, cfg, jnp.asarray(xl, dtype), jnp.asarray(dl, dtype)))
    dh = tgt.distance_in(xl[out], dl[out])
    fin = np.isfinite(dh)
    assert np.array_equal(np.isfinite(di[out]), fin) and np.allclose(di[out][fin], dh[fin], rtol=1e-4, atol=1e-3)
    prim = C.primaries(3000, tgt, np.random.default_rng(3))
    hr = C.run(bank, tgt, lambda x: np.tile([0.0, 0.0, 2.0], (len(x), 1)), env, prim, np.random.default_rng(4))
    rec, cnt = J.run(bank, tgt, fm, env, prim, seed=5, pool=4096, steps=32, dtype=dtype)
    assert cnt["inelastic"] == hr["counts"]["inelastic"] == 3000
    nh, nd = hr["counts"]["recorded"], len(rec["pdg"])
    assert nh > 1500 and abs(nd - nh) < 4 * np.sqrt(nh), (nh, nd)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_tally_planes_record_without_changing_transport(tmp_path, dtype):
    """Recording-only planes: the record plane and counters are unchanged (no random numbers drawn), crossings in
    both directions sit on their planes, and device and host agree per plane and direction within statistics."""
    import dataclasses
    bank = _bank(tmp_path, n_pim=3)
    tgt = D.Target.rod(100.0, 10.0, 10.0, 10.0)
    env0 = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                      z_plane=800.0, r_plane=1800.0, axis_x=0.0)
    planes = ((-300.0, 1800.0), (300.0, 1800.0), (600.0, 900.0))
    env = dataclasses.replace(env0, tally=planes)
    prim = C.primaries(4000, tgt, np.random.default_rng(3))
    rec0, cnt0 = J.run(bank, tgt, _uniform_map(2.0), env0, prim, seed=5, pool=4096, steps=32, dtype=dtype)
    rec, cnt = J.run(bank, tgt, _uniform_map(2.0), env, prim, seed=5, pool=4096, steps=32, dtype=dtype)
    tal = rec.pop("tally")
    for k in rec0:
        assert np.array_equal(rec0[k], rec[k]), k
    assert {k: v for k, v in cnt.items() if k not in ("tallied",)} == {k: v for k, v in cnt0.items() if k != "tallied"}
    assert cnt["tallied"] == len(tal["pdg"]) > 0
    bfield = lambda x: np.tile([0.0, 0.0, 2.0], (len(x), 1))
    h0 = C.run(bank, tgt, bfield, env0, prim, np.random.default_rng(4))
    h = C.run(bank, tgt, bfield, env, prim, np.random.default_rng(4))
    for k in h0["record"]:
        assert np.array_equal(h0["record"][k], h["record"][k]), k
    ht = h["tally"]
    for t in (tal, ht):
        assert set(np.unique(t["pdg"])) <= {13, -13, 211, -211}
        z = np.asarray(planes)[t["plane"], 0]
        assert np.allclose(t["x"][:, 2], z, atol=1e-2)
        assert np.all(np.hypot(t["x"][:, 0], t["x"][:, 1]) < np.asarray(planes)[t["plane"], 1])
    for kp in range(len(planes)):
        for sgn in (1, -1):
            nd = int(np.sum((tal["plane"] == kp) & (np.sign(tal["p"][:, 2]) == sgn)))
            nh = int(np.sum((ht["plane"] == kp) & (np.sign(ht["p"][:, 2]) == sgn)))
            assert abs(nd - nh) < 4 * np.sqrt(max(nh, 1)) + 2, (kp, sgn, nd, nh)
    assert np.sum((ht["plane"] == 0) & (ht["p"][:, 2] < 0)) > 200       # upstream plane sees backward lanes
    assert np.sum((ht["plane"] == 1) & (ht["p"][:, 2] > 0)) > 1000


def test_tally_planes_must_be_a_vacuum_step_apart():
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0, tally=((790.0, 100.0),))
    with pytest.raises(ValueError, match="vacuum step"):
        C.check_tally(env, C.Stepping().h_vac)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_target_heat_slices_device_matches_host(tmp_path, dtype):
    """Slice deposits (ionisation at step midpoints, stopped lanes, local deposit per interaction): the host's slice
    sums equal its per-segment sums, and device and host agree per slice within statistics."""
    bank = _bank(tmp_path, n_pim=3, sigma_p=5e-3, dedx=0.5, edep_local=40.0)
    tgt = D.Target.rod(100.0, 10.0, 10.0, 10.0)
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0)
    prim = C.primaries(4000, tgt, np.random.default_rng(3))
    host = C.run(bank, tgt, lambda x: np.tile([0.0, 0.0, 2.0], (len(x), 1)), env, prim, np.random.default_rng(4))
    hz = host["edep_slices"]
    assert np.isclose(hz.sum(), host["edep"].sum(), rtol=1e-9)
    ez, n = D.slice_edges(tgt)
    assert n == 10 and np.allclose(hz[:, n:], 0.0)
    rec, cnt = J.run(bank, tgt, _uniform_map(2.0), env, prim, seed=5, pool=4096, steps=32, dtype=dtype)
    dz = rec["edep_slices"]
    assert dz.shape == (D.EDEP_BATCHES, D.EDEP_SLICE_MAX) and np.allclose(dz[:, n:], 0.0)
    h, d = hz.sum(0)[:n], dz.sum(0)[:n]
    assert np.all(h > 0)
    sig = np.sqrt(np.var(hz[:, :n] * D.EDEP_BATCHES, axis=0, ddof=1) / D.EDEP_BATCHES)   # batch-mean error
    assert np.all(np.abs(d - h) < 5 * np.sqrt(2) * sig + 1e-6 * h), (h, d, sig)
    assert abs(d.sum() / h.sum() - 1) < 0.03


def test_slice_masses_and_peak_heat():
    tgt = D.Target.rod(160.0, 3.0, 3.5, 4.0)
    ez, n = D.slice_edges(tgt)
    m = D.slice_masses(tgt, ez, [19.3])
    r = lambda z: np.interp(z, [-80.0, 0.0, 80.0], [3.0, 3.5, 4.0])
    exact = np.array([np.pi / 3 * (ez[k + 1] - ez[k]) * (r(ez[k]) ** 2 + r(ez[k]) * r(ez[k + 1]) + r(ez[k + 1]) ** 2)
                      for k in range(n)]) * 19.3e-3
    assert n == 16 and np.allclose(m[:n], exact, rtol=0.01) and np.all(m[n:] == 0)
    b = np.zeros((D.EDEP_BATCHES, D.EDEP_SLICE_MAX)); b[:, 3] = 100.0; b[:, 4] = 50.0; b[0, 3] = 110.0
    h = D.peak_heat(b, m, 1000.0, n)
    assert h["peak_slice"] == 3
    assert np.isclose(h["peak_MeV_per_g_per_pot"], b[:, 3].sum() / m[3] / 1000.0)
    assert np.isclose(h["peak_Gy_per_pot"], h["peak_MeV_per_g_per_pot"] * D.MEV_PER_G_TO_GY) and h["peak_err"] > 0
    assert D.MEV_PER_G_TO_GY == pytest.approx(1.602176634e-10)          # 1 MeV/g in Gy
    with pytest.raises(ValueError, match="slices"):
        D.slice_edges(D.Target.rod(1000.0, 3.0, 3.0, 3.0))
    with pytest.raises(ValueError, match="density"):
        D.material_densities(["unobtainium"])


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_device_rotated_boxes_match_host(tmp_path, dtype):
    """Stickman-like fins at 285, 165 and 45 degrees around a core: device and host locate, distances and the
    cascade agree."""
    bank = _bank(tmp_path, n_pim=3)
    ang = np.deg2rad([285.0, 165.0, 45.0])
    fins = np.array([[0.0, 18.0, -1.0, 1.0, -30.0, 30.0]] * 3)
    tgt = D.Target(np.array([-30.0, 30.0]), np.zeros((1, 2)), np.full((1, 2), 3.0), np.zeros(4, int), boxes=fins,
                   box_phi=ang)
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([1900.0, 1900.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=1800.0, axis_x=0.0)
    fm = _uniform_map(2.0)
    S, nk, nmat = J.make_static(bank, tgt, env, fm, dtype)
    cfg = J.make_cfg(tgt, env, C.Stepping(), nk, nmat, dtype)
    rng = np.random.default_rng(11)
    xl = np.stack([rng.uniform(-20, 20, 20000), rng.uniform(-20, 20, 20000), rng.uniform(-35, 35, 20000)], 1)
    host = tgt.locate(xl)
    dev = np.asarray(J.locate(S, cfg, jnp.asarray(xl, dtype)))
    assert np.mean(dev == host) > 0.9995 and set(host.tolist()) == {-1, 0, 1, 2, 3}
    same = dev == host
    dl = rng.normal(size=xl.shape); dl /= np.linalg.norm(dl, axis=1, keepdims=True)
    x, d = jnp.asarray(xl, dtype), jnp.asarray(dl, dtype)
    inside = same & (host >= 0)
    do = np.asarray(J.distance_out(S, cfg, x, d, jnp.asarray(host, jnp.int32)))[inside]
    assert np.mean(np.isclose(do, tgt.distance_out(xl[inside], dl[inside], host[inside]), rtol=1e-4, atol=1e-3)) > 0.999
    out = same & (host < 0)
    di, dh = np.asarray(J.distance_in(S, cfg, x, d))[out], tgt.distance_in(xl[out], dl[out])
    fin = np.isfinite(dh)
    assert np.mean(np.isfinite(di) == fin) > 0.999 and fin.sum() > 1000
    both = fin & np.isfinite(di)
    assert np.mean(np.isclose(di[both], dh[both], rtol=1e-4, atol=1e-3)) > 0.999
    prim = C.primaries(3000, tgt, np.random.default_rng(3))
    hr = C.run(bank, tgt, lambda q: np.tile([0.0, 0.0, 2.0], (len(q), 1)), env, prim, np.random.default_rng(4))
    rec, cnt = J.run(bank, tgt, fm, env, prim, seed=5, pool=4096, steps=32, dtype=dtype)
    assert cnt["inelastic"] == hr["counts"]["inelastic"] == 3000
    nh, nd = hr["counts"]["recorded"], len(rec["pdg"])
    assert nh > 1500 and abs(nd - nh) < 4 * np.sqrt(nh), (nh, nd)
