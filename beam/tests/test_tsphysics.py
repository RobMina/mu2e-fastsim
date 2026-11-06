"""Synthetic TS primitives: no field maps, Geant4, JAX or staged beam inputs."""
import numpy as np
import pytest

from mu2efastsim import eloss, materials, transport
from beamfastsim import tsphysics as physics


def _rng(seed=1234):
    return np.random.default_rng(seed)


def _p_from_t(kinetic, mass):
    return np.sqrt(kinetic * (kinetic + 2 * mass))


@pytest.mark.parametrize("boost", [0.0, 0.01, 2.0, 1e3, 1e5])
def test_decay_four_momentum_conservation(boost):
    rng = _rng()
    direction = rng.normal(size=(4096, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    parent = direction * (boost * physics.MPI)
    original = parent.copy()
    mu, nu = physics.decay_pions(rng, parent)
    emu = np.hypot(np.linalg.norm(mu, axis=1), physics.MMU)
    enu = np.linalg.norm(nu, axis=1)
    epi = np.hypot(np.linalg.norm(parent, axis=1), physics.MPI)
    np.testing.assert_allclose(emu + enu, epi, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(mu + nu, parent, rtol=1e-13, atol=1e-8)
    if boost <= 1e3:
        invariant = emu ** 2 - np.sum(mu * mu, axis=1)
        np.testing.assert_allclose(invariant, physics.MMU ** 2, rtol=3e-9, atol=1e-9)
    np.testing.assert_array_equal(parent, original)


def test_decay_rest_frame_spectrum_is_isotropic():
    mu, nu = physics.decay_pions(_rng(), np.zeros((100_000, 3)))
    pstar = (physics.MPI ** 2 - physics.MMU ** 2) / (2 * physics.MPI)
    np.testing.assert_allclose(np.linalg.norm(mu, axis=1), pstar, rtol=1e-14)
    np.testing.assert_array_equal(nu, -mu)
    direction = mu / pstar
    np.testing.assert_allclose(direction.mean(axis=0), 0.0, atol=0.006)
    np.testing.assert_allclose(direction.T @ direction / len(mu), np.eye(3) / 3, atol=0.005)
    assert np.mean(direction[:, 2] > 0.5) == pytest.approx(0.25, abs=0.005)


def test_exact_constants_and_daughter_charge_conventions():
    assert physics.MPI == 139.57039
    assert physics.MMU == 105.6583755 == eloss.MMU
    assert physics.TAU_PI_NS == 26.033
    assert physics.TAU_MU_NS == 2196.9811
    pdg_parent = np.array([211, -211])
    pdg_mu = np.array([-13, 13])
    pdg_nu = np.array([14, -14])
    np.testing.assert_array_equal(np.sign(pdg_parent), -np.sign(pdg_mu))
    np.testing.assert_array_equal(np.sign(pdg_parent), np.sign(pdg_nu))
    np.testing.assert_array_equal(physics.mass_mev(pdg_parent), [physics.MPI] * 2)
    np.testing.assert_array_equal(physics.mass_mev(pdg_mu), [physics.MMU] * 2)
    np.testing.assert_array_equal(physics.lifetime_ns([211, -211, 13, -13]),
                                  [26.033, 26.033, 2196.9811, 2196.9811])


@pytest.mark.parametrize("pdg,tau", [(211, 26.033), (-211, 26.033), (13, 2196.9811), (-13, 2196.9811)])
def test_decay_lifetime_rates_and_time_dilation(pdg, tau):
    proper = physics.sample_proper_lifetime(_rng(), np.full(100_000, pdg))
    assert np.all(proper >= 0)
    assert proper.mean() == pytest.approx(tau, rel=0.015)
    assert np.mean(proper > tau) == pytest.approx(np.exp(-1), abs=0.005)
    gamma = 7.0
    lab = gamma * proper
    assert np.mean(lab > tau) == pytest.approx(np.exp(-1 / gamma), abs=0.005)
    assert np.mean(lab > gamma * tau) == pytest.approx(np.exp(-1), abs=0.005)
    distance = np.sqrt(gamma ** 2 - 1) * physics.C_MM_NS * proper
    mean_length = np.sqrt(gamma ** 2 - 1) * physics.C_MM_NS * tau
    assert np.mean(distance > mean_length) == pytest.approx(np.exp(-1), abs=0.005)


def test_lifetime_scalar_and_charge_symmetry():
    assert np.ndim(physics.sample_proper_lifetime(_rng(), 211)) == 0
    pdg = np.tile([211, 13], 100)
    np.testing.assert_array_equal(physics.sample_proper_lifetime(_rng(), pdg),
                                  physics.sample_proper_lifetime(_rng(), -pdg))


@pytest.mark.parametrize("mat", [materials.ALUMINIUM, materials.POLYETHYLENE, materials.TUNGSTEN])
def test_material_charge_symmetry_and_energy_nonincrease(mat):
    rng = _rng()
    pdg = np.tile([211, 13], 512)
    pvec = rng.normal(size=(len(pdg), 3)) * 100
    length = rng.uniform(0.001, 0.05, len(pdg))
    previous = rng.uniform(0, 0.2, len(pdg))
    original = pvec.copy()
    plus = physics.material_step(_rng(), mat, pvec, pdg, length, previous)
    minus = physics.material_step(_rng(), mat, pvec, -pdg, length, previous)
    for a, b in zip(plus, minus):
        np.testing.assert_array_equal(a, b)
    after, fraction, stopped = plus
    assert np.all(np.isfinite(after))
    assert np.all(np.linalg.norm(after, axis=1) <= np.linalg.norm(pvec, axis=1) + 1e-12)
    assert np.all((fraction >= 0) & (fraction <= 1))
    assert stopped.dtype == bool
    np.testing.assert_array_equal(after[stopped], 0.0)
    np.testing.assert_array_equal(pvec, original)


def test_mass_arrays_are_forwarded_to_core(monkeypatch):
    calls = []

    def mean(mat, p, cut, mass):
        calls.append(("mean", mass.copy()))
        return np.full_like(p, 0.1)

    def sample(rng, mat, p, length, cut, mass):
        calls.append(("sample", mass.copy()))
        return np.full_like(p, 0.1) * length

    def variance(p, previous, new, mass):
        calls.append(("scatter", mass.copy()))
        return np.zeros_like(p)

    monkeypatch.setattr(eloss, "dedx_restricted", mean)
    monkeypatch.setattr(eloss, "sample_loss", sample)
    monkeypatch.setattr(transport, "highland_var_increment", variance)
    pdg = np.array([211, -211, 13, -13])
    pvec = np.tile([100.0, 0.0, 0.0], (4, 1))
    after, fraction, stopped = physics.material_step(_rng(), materials.ALUMINIUM, pvec, pdg, 1)
    assert [name for name, _ in calls] == ["mean", "sample", "scatter"]
    for _, mass in calls:
        np.testing.assert_array_equal(mass, [physics.MPI, physics.MPI, physics.MMU, physics.MMU])
    initial_t = np.hypot(100, physics.mass_mev(pdg)) - physics.mass_mev(pdg)
    final_t = np.hypot(np.linalg.norm(after, axis=1), physics.mass_mev(pdg)) - physics.mass_mev(pdg)
    np.testing.assert_allclose(initial_t - final_t, 0.1, atol=5e-14)
    np.testing.assert_array_equal(fraction, 1.0)
    assert not stopped.any()


def test_rangeout_fraction_and_unphysical_tail_fallback(monkeypatch):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut, mass: np.full_like(p, 2.0))
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut, mass: np.full_like(p, 1e6))
    pdg = np.array([211, -211, 13, -13, 211])
    kinetic = np.array([0.0, 0.1, 1.0, 10.0, 10.0])
    pvec = np.zeros((5, 3))
    pvec[:, 2] = _p_from_t(kinetic, physics.mass_mev(pdg))
    after, fraction, stopped = physics.material_step(
        _rng(), materials.ALUMINIUM, pvec, pdg, [3.0, 3.0, 1.0, 10.0, 0.0])
    np.testing.assert_array_equal(stopped, [True, True, True, True, False])
    np.testing.assert_allclose(fraction, [0.0, 0.0, 0.4, 0.49, 1.0], atol=1e-14)
    np.testing.assert_array_equal(after[:4], 0.0)
    np.testing.assert_array_equal(after[4], pvec[4])


@pytest.mark.parametrize("mat", [materials.ALUMINIUM, materials.POLYETHYLENE, materials.TUNGSTEN])
def test_real_core_loss_ranges_out_both_species(mat):
    pdg = np.array([211, -211, 13, -13])
    mass = physics.mass_mev(pdg)
    pvec = np.zeros((4, 3))
    pvec[:, 2] = _p_from_t(np.full(4, 2.0), mass)
    after, fraction, stopped = physics.material_step(_rng(), mat, pvec, pdg, 100.0)
    assert stopped.all()
    assert np.all((fraction > 0) & (fraction < 1))
    np.testing.assert_array_equal(after, 0.0)
    mean = eloss.dedx_restricted(mat, pvec[:, 2], 1e9, mass=mass) * 100.0
    np.testing.assert_allclose(fraction, (2.0 - physics.TK_STOP_MEV) / mean)


def test_sampled_threshold_crossing(monkeypatch):
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut, mass: np.full_like(p, 0.85))
    pvec = np.array([[0.0, 0.0, _p_from_t(1.0, physics.MPI)]])
    after, fraction, stopped = physics.material_step(_rng(), materials.ALUMINIUM, pvec, 211, 1.0)
    assert stopped[0]
    assert fraction[0] == pytest.approx(0.8 / 0.85)
    np.testing.assert_array_equal(after, 0.0)


def test_zero_step_is_identity_without_random_draws():
    pvec = np.array([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]])
    rng, control = _rng(), _rng()
    after, fraction, stopped = physics.material_step(rng, materials.ALUMINIUM, pvec, [211, 13], 0.0)
    np.testing.assert_array_equal(after, pvec)
    np.testing.assert_array_equal(fraction, [1.0, 0.0])
    np.testing.assert_array_equal(stopped, [False, True])
    assert rng.random() == control.random()


@pytest.mark.parametrize("pdg,previous", [(211, 0.0), (13, 0.03)])
def test_scatter_normalization_and_core_variance(monkeypatch, pdg, previous):
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut, mass: np.zeros_like(p))
    n, p, length = 100_000, 1000.0, 1.0
    pvec = np.tile([0.0, 0.0, p], (n, 1))
    after, fraction, stopped = physics.material_step(
        _rng(), materials.ALUMINIUM, pvec, pdg, length, previous)
    np.testing.assert_allclose(np.linalg.norm(after, axis=1), p, rtol=5e-16)
    slopes = after[:, :2] / after[:, [2]]
    x0 = eloss.radiation_length_mm(materials.ALUMINIUM)
    expected = transport.highland_var_increment(p, previous, previous + length / x0, mass=physics.mass_mev(pdg))
    np.testing.assert_allclose(slopes.var(axis=0), expected, rtol=0.02)
    np.testing.assert_allclose(slopes.mean(axis=0), 0, atol=np.sqrt(expected) * 0.012)
    assert abs(np.mean(slopes[:, 0] * slopes[:, 1])) < 0.012 * expected
    np.testing.assert_array_equal(fraction, 1.0)
    assert not stopped.any()


def test_empty_batches():
    mu, nu = physics.decay_pions(_rng(), np.empty((0, 3)))
    assert mu.shape == nu.shape == (0, 3)
    assert physics.sample_proper_lifetime(_rng(), []).shape == (0,)
    after, fraction, stopped = physics.material_step(_rng(), materials.ALUMINIUM, np.empty((0, 3)), [], [])
    assert after.shape == (0, 3)
    assert fraction.shape == stopped.shape == (0,)


@pytest.mark.parametrize("pdg", [0, 11, 14, 22, 211.5, np.nan, np.inf, [211, 2212]])
def test_invalid_species_rejected(pdg):
    with pytest.raises(ValueError, match="pdg"):
        physics.mass_mev(pdg)
    with pytest.raises(ValueError, match="pdg"):
        physics.sample_proper_lifetime(_rng(), pdg)
    with pytest.raises(ValueError, match="pdg"):
        physics.material_step(_rng(), materials.ALUMINIUM, np.ones((2, 3)), pdg, 0.0)


@pytest.mark.parametrize("pvec", [[1.0, 2.0, 3.0], np.ones((2, 4)), [[np.nan, 0, 0]], [[np.inf, 0, 0]]])
def test_invalid_momenta_rejected(pvec):
    with pytest.raises(ValueError, match="pvec"):
        physics.decay_pions(_rng(), pvec)
    with pytest.raises(ValueError, match="pvec"):
        physics.material_step(_rng(), materials.ALUMINIUM, pvec, 211, 1.0)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, [1.0, 2.0, 3.0]])
@pytest.mark.parametrize("name", ["length_mm", "thickness_x0_before"])
def test_invalid_lengths_rejected(name, value):
    kwargs = {"length_mm": 1.0, "thickness_x0_before": 0.0, name: value}
    with pytest.raises(ValueError, match=name):
        physics.material_step(_rng(), materials.ALUMINIUM, np.ones((2, 3)), 211, **kwargs)


def test_invalid_material_rejected():
    with pytest.raises(ValueError, match="mat"):
        physics.material_step(_rng(), "G4_Al", np.ones((2, 3)), 211, 1.0)
