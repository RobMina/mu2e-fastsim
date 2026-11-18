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
    env = C.Envelope(z_nodes=np.array([-5000.0, 5000.0]), r_nodes=np.array([400.0, 400.0]), z_min=-2000.0,
                     z_plane=800.0, r_plane=300.0, axis_x=0.0)
    fm = _uniform_map(2.0)
    rng = np.random.default_rng(3)
    n = 4000
    prim = C.primaries(n, tgt, rng)
    host = C.run(bank, tgt, lambda x: np.tile([0.0, 0.0, 2.0], (len(x), 1)), env, prim, np.random.default_rng(4))
    rec, cnt = J.run(bank, tgt, fm, env, prim, seed=5, pool=4096, steps=32, dtype=dtype)
    assert cnt["inelastic"] == host["counts"]["inelastic"] == n          # every proton interacts once
    nh, nd = host["counts"]["recorded"], len(rec["pdg"])
    assert nh > 200
    assert abs(nd - nh) < 4 * np.sqrt(nh)                                  # same physics, independent streams
    assert set(np.unique(rec["pdg"])) <= {13, -13, 211, -211}
    assert np.allclose(rec["x"][:, 2], 800.0, atol=1e-3)
    ph, pd = np.linalg.norm(host["record"]["p"], axis=1), np.linalg.norm(rec["p"], axis=1)
    assert abs(np.median(ph) - np.median(pd)) < 0.1 * np.median(ph)


def test_refill_moves_emissions_into_dead_lanes():
    N, E = 8, 16
    L = J.Lanes(alive=jnp.array([True, False] * 4), sp=jnp.zeros(N, jnp.int32), x=jnp.zeros((N, 3)), u=jnp.zeros((N, 3)),
                p=jnp.ones(N), t=jnp.zeros(N), tau=jnp.full(N, jnp.inf), seg=jnp.full(N, -1, jnp.int32), ev=jnp.zeros(N, jnp.int64))
    emit = J._empty_emit(E, jnp.float64)
    emit = (emit[0], emit[1].at[:6].set(J.species_of(np.array([211] * 6))), emit[2], emit[3], emit[4].at[:6].set(7.0),
            emit[5], emit[6], emit[7].at[:6].set(jnp.arange(6)))
    L2, emit2, left = J._refill(L, emit, jnp.int64(6), jax.random.PRNGKey(0), 1.0)
    assert int(jnp.sum(L2.alive)) == 8 and int(left) == 2
    assert np.allclose(np.asarray(L2.p)[1::2], 7.0) and list(np.asarray(emit2[7][:2])) == [4, 5]
    assert np.all(np.isfinite(np.asarray(L2.tau)[1::2]))                   # pions get a lifetime
