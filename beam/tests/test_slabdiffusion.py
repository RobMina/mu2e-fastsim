"""Numerical-contract checks, not Geant4 closure or tail qualification."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mu2efastsim import eloss_jax, materials
from beamfastsim import slabdiffusion as sd


def tables(lam=0.3, slope=1.0):
    e = np.geomspace(0.1, 100., 41)
    return sd.make_tables(e, slope * e, e, np.full_like(e, lam))


def directions(n, cosine=1.):
    return np.tile([np.sqrt(1 - cosine ** 2), 0., cosine], (n, 1))


def same(a, b):
    for field in sd.Response._fields[:-1]:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert a.ledger.keys() == b.ledger.keys()
    for name in a.ledger:
        np.testing.assert_array_equal(a.ledger[name], b.ledger[name])


def test_straight_both_faces_and_analytic_mean():
    u = np.array([[0., 0., 1.], [0.6, 0., 0.8], [0.6, 0., -0.8]])
    r = sd.respond(np.full(3, 5.), u, 0.4, tables(), scatter=False, loss="mean", max_step=.07)
    length = .4 / np.abs(u[:, 2])
    np.testing.assert_allclose(r.path[0], length, atol=1e-14)
    np.testing.assert_allclose(r.raw_energy[0], 5. - length, atol=1e-13)
    np.testing.assert_allclose(r.position[0, :, 0], length * u[:, 0], atol=1e-14)
    np.testing.assert_array_equal(r.position[0, :, 2], [.4, .4, 0.])
    np.testing.assert_array_equal(r.direction[0], u)
    assert np.all(r.status == sd.TRANSMITTED)
    np.testing.assert_allclose(r.ledger["applied_loss"] + r.raw_energy, 5., atol=1e-14)
    assert np.all(r.ledger["angular_time"] == 0)


def test_non_linear_range_mean_and_no_loss_control():
    e = np.geomspace(.1, 100, 51)
    tb = sd.make_tables(e, e**2, e, np.ones_like(e))
    r = sd.respond(np.array([3.]), directions(1), 2., tb, scatter=False, loss="mean", max_step=.3)
    np.testing.assert_allclose(r.raw_energy, np.sqrt(7), atol=2e-14)
    r = sd.respond(np.array([3.]), directions(1, .8), .4, tb, scatter=False, loss="none")
    assert r.raw_energy[0, 0] == 3.
    np.testing.assert_allclose(r.path, .5, atol=1e-14)
    assert not np.any(r.ledger["applied_loss"])


def test_reflection_and_bounded_geometry():
    n = 1200
    r = sd.respond(np.full(n, 5.), directions(n), .5, tables(lam=.08), loss="none",
                   max_step=.02, angular_fraction=.05, max_steps=4000, seed=33, batch=n)
    assert np.count_nonzero(r.status == sd.REFLECTED) > n // 3
    assert np.count_nonzero(r.status == sd.TRANSMITTED) > 10
    assert not np.any(r.status == sd.UNRESOLVED_NUMERICAL)
    assert np.all((r.position[..., 2] >= 0) & (r.position[..., 2] <= .5))
    np.testing.assert_allclose(np.linalg.norm(r.direction, axis=-1), 1, atol=3e-15)
    exited = (r.status == sd.TRANSMITTED) | (r.status == sd.REFLECTED)
    assert np.all(r.path[r.status == sd.TRANSMITTED] >= .5)
    assert np.all(r.position[..., 2][r.status == sd.REFLECTED] == 0)
    assert np.all(r.direction[..., 2][r.status == sd.REFLECTED] < 0)
    assert np.all(r.direction[..., 2][r.status == sd.TRANSMITTED] > 0)
    assert np.all(r.ledger["face_reached"][exited])
    assert np.all(np.linalg.norm(r.position, axis=-1) <= r.path + 1e-13)
    assert np.all(r.steps <= 4000)


def test_exact_batch_independence_sample_loss_and_separate_calls():
    n = 7
    e = np.linspace(1., 2., n)
    u = directions(n, .8)
    rows = np.arange(100, 100 + n, dtype=np.uint32)
    kw = dict(material=materials.POLYSTYRENE, replicas=2, seed=117, max_steps=300,
              max_step=.02, angular_fraction=.07, row_ids=rows, replica_offset=9)
    a = sd.respond(e, u, .06, tables(), batch=14, **kw)
    b = sd.respond(e, u, .06, tables(), batch=1, **kw)
    c = sd.respond(e, u, .06, tables(), batch=5, **kw)
    same(a, b)
    same(a, c)
    kw.update(row_ids=rows[3:5], replicas=1, replica_offset=10)
    d = sd.respond(e[3:5], u[3:5], .06, tables(), batch=2, **kw)
    for field in sd.Response._fields[:-1]:
        np.testing.assert_array_equal(getattr(d, field), getattr(a, field)[1:2, 3:5])
    for name in d.ledger:
        np.testing.assert_array_equal(d.ledger[name], a.ledger[name][1:2, 3:5])
    np.testing.assert_allclose(a.raw_energy + a.ledger["applied_loss"], e[None, :] + np.zeros((2, 1)), atol=1e-14)
    np.testing.assert_allclose(a.ledger["requested_loss"], a.ledger["applied_loss"] + a.ledger["clipped_loss"], atol=1e-14)
    assert np.any(a.ledger["requested_loss"] != a.ledger["mean_loss"])


def test_angular_constant_energy_moments():
    # Constant lambda=1, total true path s=.6. P_l moments approach
    # exp(-l(l+1)s/2); two transverse components EACH have variance h/lambda.
    n, steps, q = 30000, 60, .01
    keys = jax.vmap(lambda row: jax.random.fold_in(jax.random.key(81), row))(jnp.arange(n, dtype=jnp.uint32))

    @jax.jit
    def run():
        def step(i, u):
            k = jax.vmap(lambda key: jax.random.fold_in(key, i))(keys)
            return jax.vmap(sd.angular_step)(k, u, jnp.full(n, q))
        return jax.lax.fori_loop(0, steps, step, jnp.asarray(directions(n)))

    u = np.asarray(run())
    z = u[:, 2]
    for measured, expected in ((z, np.exp(-.6)), ((3 * z*z - 1)/2, np.exp(-1.8))):
        # Sampling uncertainty plus explicit O(q) finite-step bias allowance.
        assert abs(measured.mean() - expected) < 5 * measured.std(ddof=1) / np.sqrt(n) + .003
    assert abs(u[:, 0].mean()) < .01 and abs(u[:, 1].mean()) < .01
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1., atol=3e-15)
    np.testing.assert_array_equal(sd.angular_step(jax.random.key(1), jnp.array([0., 0., 1.]), 0.), [0., 0., 1.])


def test_floor_is_unresolved_not_zero_and_table_domain_not_extrapolated():
    r = sd.respond(np.array([.05, .1, .2]), directions(3), 1., tables(), scatter=False, loss="mean", max_step=.3)
    assert np.all(r.status == sd.UNRESOLVED_FLOOR)
    np.testing.assert_allclose(r.raw_energy, [[.05, .1, .1]], atol=1e-15)
    np.testing.assert_allclose(r.path, [[0., 0., .1]], atol=1e-15)
    assert np.all(r.steps[0, :2] == 0)
    assert np.all(r.ledger["floor_reached"])
    assert not np.any(r.ledger["rawzero"])
    assert not np.any(r.ledger["face_reached"])


def test_empty_zero_cap_and_tangent_cap():
    empty = sd.respond(np.empty(0), np.empty((0, 3)), .1, tables(), loss="mean", replicas=3)
    assert empty.raw_energy.shape == (3, 0) and empty.position.shape == (3, 0, 3)
    assert all(v.shape == (3, 0) for v in empty.ledger.values())
    r = sd.respond(np.array([1.]), directions(1), .1, tables(), loss="none", max_steps=0)
    assert r.status[0, 0] == sd.UNRESOLVED_CAP and r.steps[0, 0] == 0
    assert r.raw_energy[0, 0] == 1. and r.path[0, 0] == 0.
    r = sd.respond(np.array([1.]), np.array([[1., 0., 0.]]), .1, tables(),
                   scatter=False, loss="none", max_steps=3, max_step=.01)
    assert r.status[0, 0] == sd.UNRESOLVED_CAP and r.steps[0, 0] == 3
    np.testing.assert_allclose(r.position, [[[.03, 0., 0.]]])
    np.testing.assert_allclose(r.path, .03)


def test_sample_api_mean_rawzero_and_subfloor_exit(monkeypatch):
    calls = []

    def controlled(key, mat, p, length, tcut, *, mean):
        calls.append(mean)
        # Controlled sample: crossing 0.005 mm loses .15 MeV, an interior
        # 0.01 mm leg loses .30 MeV. Input T=.2 MeV.
        return 30 * length

    monkeypatch.setattr(eloss_jax, "sample_loss", controlled)
    # Distinct cfg ensures no already-compiled sampler can bypass the patch.
    r = sd.respond(np.array([.2, .2]), directions(2), np.array([.005, .5]), tables(),
                   material=materials.POLYSTYRENE, scatter=False, max_step=.010001, max_steps=123,
                   energy_fraction=.1)
    assert calls
    assert r.status[0, 0] == sd.TRANSMITTED
    np.testing.assert_allclose(r.raw_energy[0, 0], .05)
    assert r.ledger["floor_reached"][0, 0] and r.ledger["face_reached"][0, 0]
    assert r.status[0, 1] == sd.RAWZERO and r.raw_energy[0, 1] == 0
    assert r.ledger["rawzero"][0, 1] and r.ledger["clipped_loss"][0, 1] > .1
    assert not r.ledger["face_reached"][0, 1]
    np.testing.assert_allclose(r.ledger["mean_loss"], [[.005, .010001]], atol=1e-15)
    np.testing.assert_allclose(r.raw_energy + r.ledger["applied_loss"], .2)


@pytest.mark.parametrize("kw", [dict(max_step=np.nan), dict(angular_fraction=0), dict(energy_fraction=1),
                                   dict(max_steps=1.5), dict(batch=0), dict(replicas=0), dict(seed=-1),
                                   dict(floor=.01), dict(floor=200), dict(tcut=np.inf), dict(scatter=2),
                                   dict(row_ids=[-1]), dict(loss="bad")])
def test_invalid_controls(kw):
    args = dict(loss="mean")
    args.update(kw)
    with pytest.raises(ValueError):
        sd.respond(np.array([1.]), directions(1), .1, tables(), **args)


@pytest.mark.parametrize("e,u,thickness", [([np.nan], [[0, 0, 1]], .1), ([np.inf], [[0, 0, 1]], .1),
                                           ([0], [[0, 0, 1]], .1), ([101], [[0, 0, 1]], .1),
                                           ([1], [[0, 0, 2]], .1), ([1], [[0, np.nan, 1]], .1),
                                           ([1], [[0, 0, 1]], np.inf), ([1], [[0, 0, 1]], 0)])
def test_invalid_arrays(e, u, thickness):
    with pytest.raises(ValueError):
        sd.respond(np.asarray(e), np.asarray(u), thickness, tables(), loss="mean")


def test_invalid_tables_and_missing_material():
    for vals in (([.1, 1], [.1, .1], [.1, 1], [1, 1]),
                 ([.1, 1], [.1, 1], [.1, 1], [1, np.inf]),
                 ([.1, 1], [.1, 1], [1, .1], [1, 1])):
        with pytest.raises(ValueError):
            sd.make_tables(*vals)
    with pytest.raises(ValueError, match="material"):
        sd.respond(np.array([1.]), directions(1), .1, tables())
    with pytest.raises(ValueError, match="row_ids"):
        sd.respond(np.ones(2), directions(2), .1, tables(), loss="mean", row_ids=[1, 1])


def test_face_floor_coincidence_and_explicit_lower_node():
    r = sd.respond(np.array([.2]), directions(1), .1, tables(), scatter=False,
                   loss="mean", max_step=.03, floor=.1)
    assert r.status[0, 0] == sd.TRANSMITTED
    assert r.position[0, 0, 2] == .1
    assert r.ledger["face_reached"][0, 0]
    np.testing.assert_allclose(r.raw_energy, .1, atol=1e-15)


def test_reverse_entry_and_permuted_row_replay():
    n = 80
    rows = np.arange(n, dtype=np.uint32) + 800
    kw = dict(loss="none", max_step=.01, angular_fraction=.08, max_steps=1600, seed=28)
    a = sd.respond(np.full(n, 2.), directions(n, -.4), .2, tables(lam=.08), row_ids=rows, **kw)
    permutation = np.arange(n)[::-1]
    b = sd.respond(np.full(n, 2.), directions(n, -.4), .2, tables(lam=.08),
                   row_ids=rows[permutation], batch=13, **kw)
    for field in sd.Response._fields[:-1]:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field)[:, permutation])
    reflected = a.status == sd.REFLECTED
    transmitted = a.status == sd.TRANSMITTED
    assert np.any(reflected) and np.any(transmitted)
    assert np.all(a.position[..., 2][reflected] == .2)
    assert np.all(a.position[..., 2][transmitted] == 0.)
    assert np.all(a.direction[..., 2][reflected] > 0.)
    assert np.all(a.direction[..., 2][transmitted] < 0.)


def test_step_limits_on_one_capped_leg():
    # R(T)=T, lambda=.2: bounds are max_step=.03, angular=.004,
    # and energy=.05, so the angular limit wins.
    r = sd.respond(np.array([1.]), directions(1), 1., tables(lam=.2), loss="mean",
                   max_step=.03, angular_fraction=.02, max_steps=1)
    np.testing.assert_allclose(r.path, .004, atol=1e-16)
    np.testing.assert_allclose(r.raw_energy, .996, atol=1e-15)
    np.testing.assert_allclose(r.ledger["angular_time"], .02, atol=1e-16)
    assert r.status[0, 0] == sd.UNRESOLVED_CAP
    r = sd.respond(np.array([1.]), directions(1), 1., tables(), loss="mean",
                   scatter=False, max_step=.03, energy_fraction=.01, max_steps=1)
    np.testing.assert_allclose(r.path, .01, atol=1e-16)


@pytest.mark.parametrize("energy", [1., .1 * (1 + 64 * np.finfo(float).eps)])
def test_nonfinite_sample_is_unresolved_without_corrupting_ledger(monkeypatch, energy):
    monkeypatch.setattr(eloss_jax, "sample_loss", lambda *a, **kw: jnp.nan)
    r = sd.respond(np.array([energy]), directions(1), .1, tables(),
                   material=materials.POLYSTYRENE, max_steps=121)
    assert r.status[0, 0] == sd.UNRESOLVED_NUMERICAL
    assert r.raw_energy[0, 0] == energy and r.path[0, 0] == 0.
    assert not r.ledger["floor_reached"][0, 0]
    assert r.steps[0, 0] == 0 and r.ledger["numerical"][0, 0]
    assert r.ledger["applied_loss"][0, 0] == 0.


def test_empty_still_validates_scalar_thickness():
    with pytest.raises(ValueError, match="thickness"):
        sd.respond(np.empty(0), np.empty((0, 3)), np.nan, tables(), loss="mean")


def test_floor_roundoff_retains_residual_without_fictitious_debit():
    tb = tables()
    floor = float(np.exp(tb.loge[0]))
    above = np.nextafter(floor, np.inf)
    outside = floor * (1 + 64 * np.finfo(float).eps)
    r = sd.respond(np.array([above, outside]), directions(2), 1., tb,
                   scatter=False, loss="mean", max_steps=0)
    np.testing.assert_array_equal(r.status, [[sd.UNRESOLVED_FLOOR, sd.UNRESOLVED_CAP]])
    np.testing.assert_array_equal(r.raw_energy, [[above, outside]])
    np.testing.assert_array_equal(r.ledger["floor_reached"], [[True, False]])
    assert not np.any(r.path) and not np.any(r.steps)
    assert not np.any(r.ledger["applied_loss"])


def test_zero_budget_away_from_floor_stays_numerical():
    # An unrepresentably small fractional step is NOT a domain-floor event.
    r = sd.respond(np.array([1.]), directions(1), 1., tables(),
                   scatter=False, loss="mean", energy_fraction=1e-20)
    assert r.status[0, 0] == sd.UNRESOLVED_NUMERICAL
    assert r.raw_energy[0, 0] == 1.
    assert not r.ledger["floor_reached"][0, 0]
    assert r.path[0, 0] == 0. and r.ledger["applied_loss"][0, 0] == 0.


@pytest.mark.parametrize("scatter", [False, True])
def test_shipped_muplus_target_floor_and_bitwise_batch_replay(scatter):
    from beamfastsim import geometry, layerphys

    tb = layerphys.g4_tables("mu+")
    floor = float(np.exp(max(tb.loge[0], tb.lam_loge[0]) if scatter else tb.loge[0]))
    n = 256
    e, u = np.full(n, 2.02), directions(n)
    kw = dict(material=geometry.STOPPING_TARGET_AL, seed=71, scatter=scatter)
    a = sd.respond(e, u, .2, tb, batch=256, **kw)
    b = sd.respond(e, u, .2, tb, batch=73, **kw)
    same(a, b)
    assert not np.any(a.ledger["numerical"])
    assert not np.any(a.ledger["cap"])
    at_floor = a.status == sd.UNRESOLVED_FLOOR
    assert np.count_nonzero(at_floor) > n // 4
    assert np.any(a.status == sd.TRANSMITTED)
    assert np.all(a.ledger["floor_reached"][at_floor])
    assert not np.any(a.ledger["face_reached"][at_floor])
    assert np.all(a.raw_energy[at_floor] > 0)
    assert np.all(a.raw_energy[at_floor] <= floor + 16 * np.finfo(float).eps * floor)
    np.testing.assert_allclose(a.raw_energy + a.ledger["applied_loss"], e[None, :],
                               rtol=0, atol=2e-14)
    np.testing.assert_allclose(a.ledger["requested_loss"],
                               a.ledger["applied_loss"] + a.ledger["clipped_loss"],
                               rtol=0, atol=2e-14)
