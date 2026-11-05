"""Standalone analytic pion-decay diagnostics; no transport or external assets."""

from dataclasses import fields

import numpy as np
import pytest

from beamfastsim import tsdecay as decay


def assert_same_products(left, right):
    for field in fields(decay.DecayProducts):
        np.testing.assert_array_equal(getattr(left, field.name), getattr(right, field.name))


def test_constants_and_conditional_scope():
    assert decay.MPI == 139.57039
    assert decay.MMU == 105.6583755
    assert decay.C_MM_NS == 299.792458
    assert decay.TAU_PI_NS == 26.033
    assert "conditional" in decay.LIMITATIONS.lower()
    assert "100%" in decay.LIMITATIONS
    assert "polarization" in decay.LIMITATIONS
    assert "hadronic" in decay.LIMITATIONS
    assert "PS/TS" in decay.LIMITATIONS


def test_rest_frame_analytic_directions_and_charges():
    ct = np.array([1.0, -1.0, 0.0, 0.0])
    phi = np.array([0.0, 8.0, 0.0, np.pi / 2])
    result = decay.decay_muon(np.zeros((4, 3)), [211, -211, 211, -211], ct, phi)
    q = (decay.MPI**2 - decay.MMU**2) / (2 * decay.MPI)
    energy = (decay.MPI**2 + decay.MMU**2) / (2 * decay.MPI)
    expected = q * np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(result.muon_p3, expected, atol=1e-13)
    np.testing.assert_allclose(result.neutrino_p3, -expected, atol=1e-13)
    np.testing.assert_allclose(result.muon_E, energy, rtol=2e-15)
    np.testing.assert_allclose(result.neutrino_E, q, rtol=2e-15)
    np.testing.assert_array_equal(result.muon_pdg, [-13, 13, -13, 13])
    np.testing.assert_array_equal(result.neutrino_pdg, [14, -14, 14, -14])


def test_four_momentum_and_mass_shells_across_boosts():
    parent = np.array([[0, 0, 0], [0, 0, 500], [0, 0, -800], [150, -230, 91],
                       [4000, -3000, 10000], [1e-180, 0, 0]], dtype=float)
    pdg = np.array([211, -211, 211, -211, 211, -211])
    result = decay.decay_muon(parent, pdg, [1, -1, 0, 0.7, -0.9, 0.3],
                              [0, 1, 2, -3, 4, 100])
    np.testing.assert_allclose(result.muon_p3 + result.neutrino_p3, parent, atol=3e-12,
                               rtol=3e-15)
    parent_E = np.hypot(np.hypot.reduce(parent, axis=-1), decay.MPI)
    np.testing.assert_allclose(result.muon_E + result.neutrino_E, parent_E, rtol=3e-15)
    muon_mass2 = result.muon_E**2 - np.sum(result.muon_p3**2, axis=-1)
    neutrino_mass2 = result.neutrino_E**2 - np.sum(result.neutrino_p3**2, axis=-1)
    np.testing.assert_allclose(muon_mass2, decay.MMU**2, atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(neutrino_mass2, 0, atol=1e-7)
    assert np.all(result.muon_E > 0)
    assert np.all(result.neutrino_E > 0)
    np.testing.assert_array_equal(result.muon_pdg, -pdg // 211 * 13)
    charge_conjugate = decay.decay_muon(parent, -pdg, [1, -1, 0, 0.7, -0.9, 0.3],
                                       [0, 1, 2, -3, 4, 100])
    np.testing.assert_array_equal(result.muon_p3, charge_conjugate.muon_p3)
    np.testing.assert_array_equal(result.neutrino_p3, charge_conjugate.neutrino_p3)
    np.testing.assert_array_equal(result.muon_pdg, -charge_conjugate.muon_pdg)


def test_collinear_boost_analytic():
    parent_p = 700.0
    gamma = np.hypot(parent_p, decay.MPI) / decay.MPI
    beta_gamma = parent_p / decay.MPI
    q = (decay.MPI**2 - decay.MMU**2) / (2 * decay.MPI)
    e = np.hypot(q, decay.MMU)
    signs = np.array([-1.0, 1.0])
    result = decay.decay_muon([0, 0, parent_p], 211, signs, 123.0)
    np.testing.assert_allclose(result.muon_p3[:, :2], 0, atol=0)
    np.testing.assert_allclose(result.muon_p3[:, 2], gamma * signs * q + beta_gamma * e)
    np.testing.assert_allclose(result.muon_E, gamma * e + beta_gamma * signs * q)
    np.testing.assert_allclose(result.neutrino_p3[:, 2], -gamma * signs * q + beta_gamma * q)
    np.testing.assert_allclose(result.neutrino_E, gamma * q - beta_gamma * signs * q)


def test_phi_periodicity_and_scalar_shape():
    result = decay.decay_muon([3, 7, -19], -211, 0.2, -2.7)
    periodic = decay.decay_muon([3, 7, -19], -211, 0.2, -2.7 + 8 * np.pi)
    np.testing.assert_allclose(result.muon_p3, periodic.muon_p3, rtol=1e-14)
    assert result.muon_p3.shape == (3,)
    assert result.muon_E.shape == ()
    assert result.muon_pdg.shape == ()


def test_batch_broadcast_and_no_input_mutation():
    parent = np.array([[[12.0, -9, 50]], [[-18, 42, 0]]])
    pdg = np.array([[211, -211, 211, -211]])
    ct = np.array([[-1.0, -0.4, 0.6, 1.0]])
    phi = np.array([[0.3], [-1.9]])
    originals = tuple(a.copy() for a in (parent, pdg, ct, phi))
    for array in (parent, pdg, ct, phi):
        array.flags.writeable = False
    result = decay.decay_muon(parent, pdg, ct, phi)
    assert result.muon_p3.shape == (2, 4, 3)
    assert result.neutrino_p3.shape == (2, 4, 3)
    assert result.muon_E.shape == (2, 4)
    assert result.muon_pdg.shape == (2, 4)
    tiled = decay.decay_muon(np.broadcast_to(parent, (2, 4, 3)),
                             np.broadcast_to(pdg, (2, 4)),
                             np.broadcast_to(ct, (2, 4)),
                             np.broadcast_to(phi, (2, 4)))
    assert_same_products(result, tiled)
    for actual, original in zip((parent, pdg, ct, phi), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 4)])
def test_empty_batches(shape):
    result = decay.decay_muon(np.empty(shape + (3,)), 211, 0, 0)
    assert result.muon_p3.shape == shape + (3,)
    assert result.muon_E.shape == shape
    assert result.neutrino_pdg.shape == shape
    rng = np.random.default_rng(52)
    control = np.random.default_rng(52)
    sampled = decay.sample(rng, np.empty(shape + (3,)), -211)
    assert sampled.neutrino_p3.shape == shape + (3,)
    assert decay.sample_lifetime(rng, shape).shape == shape
    np.testing.assert_array_equal(rng.random(8), control.random(8))
    time = decay.proper_time(np.empty(shape), 40)
    distance, mask = decay.where_decay(np.empty(shape), 40, 2)
    assert time.shape == distance.shape == mask.shape == shape


def test_proper_time_units_broadcast_and_inverse():
    path = np.array([[0.0], [decay.C_MM_NS], [2 * decay.C_MM_NS]])
    p = decay.MPI * np.array([1.0, 2.0, 4.0])
    time = decay.proper_time(path, p)
    np.testing.assert_allclose(time, np.array([[0, 0, 0], [1, 0.5, 0.25], [2, 1, 0.5]]))
    distance, within = decay.where_decay(path, p, time)
    np.testing.assert_allclose(distance, np.broadcast_to(path, (3, 3)))
    assert np.all(within)
    np.testing.assert_allclose(decay.proper_time(2 * decay.C_MM_NS, 3, mass=6), 4)
    custom, _ = decay.where_decay(0, 3, 4, mass=6)
    np.testing.assert_allclose(custom, 2 * decay.C_MM_NS)


def test_where_decay_zero_endpoint_and_unclipped_distance():
    distance, mask = decay.where_decay([0, decay.C_MM_NS, decay.C_MM_NS, 0],
                                       decay.MPI, [0, 1, 2, 1])
    np.testing.assert_array_equal(distance, decay.C_MM_NS * np.array([0, 1, 2, 1]))
    np.testing.assert_array_equal(mask, [True, True, False, False])
    assert mask.dtype == np.bool_


def test_seeded_lifetimes_match_numpy_and_theory():
    actual = decay.sample_lifetime(np.random.default_rng(103), (4, 5))
    expected = np.random.default_rng(103).exponential(decay.TAU_PI_NS, size=(4, 5))
    np.testing.assert_array_equal(actual, expected)
    lifetimes = decay.sample_lifetime(np.random.default_rng(731), (20000,))
    assert np.all(np.isfinite(lifetimes))
    assert np.all(lifetimes >= 0)
    assert abs(lifetimes.mean() / decay.TAU_PI_NS - 1) < 0.04
    probabilities = np.array([0.1, 0.5, 0.9])
    quantiles = -decay.TAU_PI_NS * np.log1p(-probabilities)
    np.testing.assert_allclose(np.quantile(lifetimes, probabilities), quantiles, rtol=0.06)
    transformed = np.sort(-np.expm1(-lifetimes / decay.TAU_PI_NS))
    ranks = np.arange(1, lifetimes.size + 1) / lifetimes.size
    assert np.max(np.abs(transformed - ranks)) < 0.025
    assert np.shape(decay.sample_lifetime(np.random.default_rng(2), ())) == ()
    assert decay.sample_lifetime(np.random.default_rng(2), 3).shape == (3,)


def test_sample_exact_rng_consumption_and_no_hidden_lifetime():
    parent = np.array([[0.0, 0, 0], [10, -30, 500], [0, 40, -10]])
    pdg = [211, -211, 211]
    rng = np.random.default_rng(883)
    control = np.random.default_rng(883)
    draws = control.random((3, 2))
    expected = decay.decay_muon(parent, pdg, 2 * draws[:, 0] - 1, 2 * np.pi * draws[:, 1])
    assert_same_products(decay.sample(rng, parent, pdg), expected)
    np.testing.assert_array_equal(rng.random(9), control.random(9))
    assert_same_products(decay.sample(np.random.default_rng(883), parent, pdg), expected)


def test_sample_broadcast_and_rest_isotropy():
    scalar_rng = np.random.default_rng(91)
    scalar = decay.sample(scalar_rng, [0, 0, 0], 211)
    assert scalar.muon_p3.shape == (3,)
    batch = decay.sample(np.random.default_rng(123), np.zeros((2, 1, 3)), [[211, -211, 211]])
    assert batch.muon_p3.shape == (2, 3, 3)
    n = 20000
    result = decay.sample(np.random.default_rng(781), np.zeros((n, 3)), 211)
    unit = result.muon_p3 / np.linalg.norm(result.muon_p3, axis=-1, keepdims=True)
    np.testing.assert_allclose(unit.mean(axis=0), 0, atol=0.03)
    np.testing.assert_allclose((unit**2).mean(axis=0), 1 / 3, atol=0.02)
    assert abs(np.mean(unit[:, 0] * unit[:, 1])) < 0.02
    assert np.all(np.isfinite(result.neutrino_E))


@pytest.mark.parametrize("kwargs", [
    {"pion_p3": [0, 1]}, {"pion_p3": 0}, {"pion_p3": np.zeros((3, 2))},
    {"pion_p3": [0, np.nan, 0]}, {"pion_p3": [0, 0, np.inf]},
    {"pion_p3": [0, 0, 1j]}, {"pion_pdg": 13}, {"pion_pdg": 0},
    {"pion_pdg": 211.5}, {"pion_pdg": np.nan}, {"pion_pdg": np.inf},
    {"cos_theta": -1.000001}, {"cos_theta": 1.000001},
    {"cos_theta": np.nan}, {"cos_theta": np.inf},
    {"phi": np.nan}, {"phi": -np.inf}, {"phi": 1j},
    {"pion_p3": np.zeros((2, 3)), "pion_pdg": [211, -211, 211]},
    {"pion_p3": np.zeros((2, 3)), "cos_theta": [0, 0, 0]},
])
def test_decay_validation(kwargs):
    arguments = dict(pion_p3=[0, 0, 0], pion_pdg=211, cos_theta=0, phi=0)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        decay.decay_muon(**arguments)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, -np.inf, 1j])
def test_path_and_remaining_validation(bad):
    with pytest.raises(ValueError):
        decay.proper_time(bad, 100)
    with pytest.raises(ValueError):
        decay.where_decay(bad, 100, 1)
    with pytest.raises(ValueError):
        decay.where_decay(1, 100, bad)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, -np.inf, 1j])
def test_momentum_and_mass_validation(bad):
    with pytest.raises(ValueError):
        decay.proper_time(0, bad)
    with pytest.raises(ValueError):
        decay.proper_time(1, 100, mass=bad)
    with pytest.raises(ValueError):
        decay.where_decay(0, bad, 0)
    with pytest.raises(ValueError):
        decay.where_decay(1, 100, 1, mass=bad)


def test_clock_incompatible_shapes():
    with pytest.raises(ValueError):
        decay.proper_time([1, 2], [10, 20, 30])
    with pytest.raises(ValueError):
        decay.where_decay([1, 2], 100, [1, 2, 3])


@pytest.mark.parametrize("parent,pdg", [([0, 0], 211), ([0, np.nan, 0], 211),
                                        ([0, 0, 0], 13), (np.zeros((2, 3)), [211] * 3)])
def test_invalid_sample_does_not_consume_rng(parent, pdg):
    rng = np.random.default_rng(99)
    control = np.random.default_rng(99)
    with pytest.raises(ValueError):
        decay.sample(rng, parent, pdg)
    np.testing.assert_array_equal(rng.random(5), control.random(5))


def test_scaled_clocks_avoid_representable_intermediate_overflow():
    assert decay.proper_time(1e-308, 1e-308) == pytest.approx(decay.MPI / decay.C_MM_NS)
    assert decay.proper_time(0, 1e-308) == 0
    distance, within = decay.where_decay(3, 1e-308, 1e308)
    assert distance == pytest.approx(decay.C_MM_NS / decay.MPI)
    assert within
    with pytest.raises(ValueError, match="finite float64"):
        decay.proper_time(1e308, 1e-308)


def test_extreme_finite_boost_is_rejected_and_supported_boundary_is_stable():
    with pytest.raises(ValueError, match="supported pion boost"):
        decay.decay_muon([1.7e308] * 3, 211, 1, 0)
    with pytest.raises(ValueError, match="supported pion boost"):
        decay.decay_muon([0, 0, 1e9], -211, 1, 0)
    gamma = 500.0
    p_over_mass = np.sqrt(gamma * gamma - 1)
    result = decay.decay_muon([0, 0, decay.MPI * p_over_mass], 211, 1, 0)
    q = (decay.MPI**2 - decay.MMU**2) / (2 * decay.MPI)
    assert result.neutrino_E == pytest.approx(q / (gamma + p_over_mass), rel=1e-9)


def test_unsupported_boost_sample_rejected_before_random_draws():
    rng = np.random.default_rng(17)
    control = np.random.default_rng(17)
    with pytest.raises(ValueError, match="supported pion boost"):
        decay.sample(rng, [1e9, 0, 0], 211)
    np.testing.assert_array_equal(rng.random(5), control.random(5))
