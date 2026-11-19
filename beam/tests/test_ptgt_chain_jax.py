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


def _bank(tmp_path, n_pim=1, sigma_p=1e3, dedx=0.0):
    shape = (len(D.SPECIES), len(MATS), EDGES.size - 1)
    cells = {k: np.zeros(shape) for k in D.CELL_ARRAYS}
    cells["x0"][:] = 1e12
    cells["sigma_inel"][0] = sigma_p
    cells["dedx"][:] = dedx
    cells["el_cos_icdf"] = np.ones(shape + (D.N_EL,))
    ncell = int(np.prod(shape))
    per = np.array([n_pim if c // (len(MATS) * (EDGES.size - 1)) == 0 else 0 for c in range(ncell)])
    trials = dict(trial_cell=np.arange(ncell), sec_offsets=np.concatenate([[0], np.cumsum(per)]), edep_local=np.zeros(ncell))
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
