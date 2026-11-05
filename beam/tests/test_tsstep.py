"""Synthetic vacuum numerical diagnostics only; no Geant4/material validation.

These tests need the matched core ehelix RK4, but not JAX or external field maps.
"""

import os
import subprocess
import sys
from dataclasses import fields

import numpy as np
import pytest

from beamfastsim import tsstep as ts
from beamfastsim.tsdecay import C_MM_NS, MMU, MPI


ZERO = np.zeros(3)
X = np.zeros((1, 3))
P = np.array([[100.0, 0, 0]])


def assert_state_equal(a, b):
    for field in fields(ts.StepResult):
        np.testing.assert_array_equal(getattr(a, field.name), getattr(b, field.name))


def test_scope_and_constants():
    assert ts.K == 0.299792458
    for text in ("vacuum", "material-veto", "step-halving", "Geant4", "monotonicity", "stateful"):
        assert text.lower() in ts.LIMITATIONS.lower()


def test_straight_batch_clocks_species_and_no_mutation():
    x = np.arange(12.0).reshape(4, 3)
    p3 = np.array([[30, 40, 0], [0, 0, 90], [-60, 0, 0], [0, -120, 0.]])
    original_x, original_p = x.copy(), p3.copy()
    h = np.array([0., 2., 3., 4.])
    time = np.array([-1., 0., 5., 8.])
    pdg = np.array([13, -13, 211, -211])
    result = ts.step(x, p3, pdg, h, ZERO, time=time)
    p = np.linalg.norm(p3, axis=1)
    m = np.array([MMU, MMU, MPI, MPI])
    np.testing.assert_allclose(result.x, x + h[:, None] * p3 / p[:, None])
    np.testing.assert_allclose(result.p3, p3, rtol=2e-15, atol=1e-14)
    np.testing.assert_array_equal(result.path, h)
    np.testing.assert_allclose(result.time, time + h * np.hypot(p, m) / (p * C_MM_NS))
    np.testing.assert_allclose(result.proper_time, h * m / (p * C_MM_NS))
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(p3, original_p)
    custom = ts.step(x, p3, pdg, h, ZERO, mass=[1., 2., 3., 4.])
    np.testing.assert_allclose(custom.proper_time, h * [1., 2., 3., 4.] / (p * C_MM_NS))
    np.testing.assert_allclose(custom.x, result.x)


def test_uniform_field_analytic_both_species_charges():
    pdg = np.array([13, -13, 211, -211])
    q = np.array([-1., 1., 1., -1.])
    x = np.zeros((4, 3))
    p3 = np.tile([80., 0, 60.], (4, 1))
    h, b, p = 3., 1.2, 100.
    result = ts.step(x, p3, pdg, h, [0, 0, b])
    theta = ts.K * q * b * h / p
    expected_p = np.column_stack((80 * np.cos(theta), -80 * np.sin(theta), np.full(4, 60.)))
    expected_x = np.column_stack((0.8 * h * np.sin(theta) / theta,
                                  -0.8 * h * (1 - np.cos(theta)) / theta,
                                  np.full(4, 0.6 * h)))
    np.testing.assert_allclose(result.x, expected_x, atol=2e-9, rtol=1e-9)
    np.testing.assert_allclose(result.p3, expected_p, atol=2e-9, rtol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(result.p3, axis=1), p, atol=3e-14)
    # Mass overrides must not change inferred charge or spatial motion.
    other = ts.step(x, p3, pdg, h, [0, 0, b], mass=1.)
    np.testing.assert_array_equal(other.p3, result.p3)
    np.testing.assert_array_equal(other.x, result.x)


def test_angle_cap_reports_actual_path_and_clocks():
    result = ts.step(np.zeros((2, 3)), np.tile([100., 0, 0], (2, 1)),
                     [13, -211], [1000., 1.], [0, 0, 2.])
    cap = 0.1 * 100 / (ts.K * 2)
    np.testing.assert_allclose(result.path, [cap, 1.])
    np.testing.assert_allclose(result.proper_time, result.path * [MMU, MPI] / (100 * C_MM_NS))


def variable_field(x):
    return np.column_stack((0.03 * np.sin(x[:, 1] / 12),
                            0.02 * np.cos(x[:, 0] / 9),
                            1.0 + 0.02 * x[:, 0] + 0.01 * x[:, 1]))


def integrate(n):
    x, p3 = X.copy(), np.array([[60., 20., 50.]])
    for _ in range(n):
        state = ts.step(x, p3, 211, 32. / n, variable_field)
        x, p3 = state.x, state.p3
    return np.r_[x.ravel(), p3.ravel()]


def test_variable_field_requires_step_halving_numerical_convergence():
    # Self-convergence, not a global bound or a physics-equivalence gate.
    reference = integrate(256)
    errors = [np.linalg.norm(integrate(n) - reference) for n in (4, 8, 16)]
    assert errors[0] > 8 * errors[1]
    assert errors[1] > 8 * errors[2]


def test_arbitrary_plane_straight_both_directions_endpoint_and_no_bracket():
    n = np.array([1., 2., 2.]) / 3
    origin = np.array([2., -1., 4.])
    x = origin + np.array([-2., 2., -20., -6.])[:, None] * n
    p3 = np.array([1., -1., 1., 1.])[:, None] * n * 100
    result = ts.plane_step(x, p3, [13, -13, 211, -211], 6., ZERO,
                           origin, n, assume_monotone=True)
    np.testing.assert_array_equal(result.within_step, [True, True, False, True])
    np.testing.assert_allclose(result.fraction[[0, 1, 3]], [1/3, 1/3, 1], atol=2e-10)
    assert np.isnan(result.fraction[2])
    np.testing.assert_allclose(result.state.x[[0, 1, 3]], np.tile(origin, (3, 1)), atol=1e-9)
    np.testing.assert_allclose(result.state.path, [2, 2, 6, 6], atol=1e-9)
    np.testing.assert_allclose(result.end.x, x + 6 * p3 / 100)
    forward = ts.plane_step(x, p3, 13, 6, ZERO, origin, n,
                            assume_monotone=True, direction=1)
    np.testing.assert_array_equal(forward.within_step, [True, False, False, True])
    backward = ts.plane_step(x, p3, 13, 6, ZERO, origin, n,
                             assume_monotone=True, direction=-1)
    np.testing.assert_array_equal(backward.within_step, [False, True, False, False])


def test_plane_trials_reintegrate_original_rk4_not_frozen_helix(monkeypatch):
    from mu2efastsim import ehelix

    calls = []
    real = ehelix.rk4

    def recorded(x, u, p, q, field, h):
        calls.append((x.copy(), np.asarray(h).copy()))
        return real(x, u, p, q, field, h)

    def forbidden(*args, **kwargs):
        raise AssertionError("a plane root must never use a frozen helix")

    monkeypatch.setattr(ehelix, "rk4", recorded)
    monkeypatch.setattr(ehelix, "helix", forbidden)
    monkeypatch.setattr(ehelix, "to_plane", forbidden)
    x, p3 = np.array([[-2., 1., 3.]]), np.array([[70., 20., 40.]])
    result = ts.plane_step(x, p3, -13, 5., variable_field,
                           ZERO, [1, 0, 0], assume_monotone=True, atol=1e-11)
    assert result.within_step[0]
    assert len(calls) > 5
    assert any(np.all(h > 0) and np.all(h < 5) for _, h in calls)
    for trial_x, _ in calls:
        np.testing.assert_array_equal(trial_x, x)
    direct = ts.step(x, p3, -13, result.state.path, variable_field)
    assert_state_equal(result.state, direct)
    assert abs(result.state.x[0, 0]) <= 1e-11
    full = ts.step(x, p3, -13, 5., variable_field)
    assert_state_equal(result.end, full)


def test_plane_endpoint_evidence_can_disprove_but_not_certify_contract():
    # A 0.1 rad turn flips a nearly tangent x-velocity; the caller's assertion
    # is contradicted by endpoint evidence and must fail closed.
    with pytest.raises(ValueError, match="monotonicity"):
        ts.plane_step([[-1., 0, 0]], [[1., -100., 0]], -13, 1000.,
                      [0, 0, 1.], ZERO, [1, 0, 0], assume_monotone=True)


def test_plane_cap_and_fraction_are_relative_to_capped_segment():
    result = ts.plane_step([[-1., 0, 0]], P, 13, 1000., [0, 0, 2.],
                           ZERO, [1, 0, 0], assume_monotone=True)
    assert result.within_step[0]
    np.testing.assert_allclose(result.end.path, 0.1 * 100 / (ts.K * 2))
    np.testing.assert_allclose(result.state.path, result.fraction * result.end.path)
    assert abs(result.state.x[0, 0]) <= 1e-9


def test_plane_clock_uses_partial_path():
    r = ts.plane_step([[-1., 0, 0]], P, 13, 4., ZERO, ZERO, [1, 0, 0],
                      time=7., assume_monotone=True)
    np.testing.assert_allclose(r.state.path, 1., atol=1e-9)
    np.testing.assert_allclose(r.state.proper_time, MMU / (100 * C_MM_NS))
    np.testing.assert_allclose(r.state.time, 7 + np.hypot(100, MMU) / (100 * C_MM_NS))


@pytest.mark.parametrize("change", [
    {"x": [0, 0, 0]}, {"p3": [100, 0, 0]}, {"x": np.zeros((2, 3))},
    {"x": [[np.nan, 0, 0]]}, {"p3": [[np.inf, 0, 0]]}, {"p3": X},
    {"p3": [[1j, 0, 0]]}, {"pdg": 11}, {"pdg": 13.5}, {"pdg": np.nan},
    {"pdg": [[13]]}, {"mass": 0}, {"mass": -1}, {"mass": np.nan},
    {"h": -1}, {"h": np.inf}, {"h": [[1]]}, {"time": np.nan},
    {"field": [0, 0]}, {"field": [[0, 0, 1]]}, {"field": [0, 0, np.nan]},
    {"field": lambda x: np.zeros(3)},
    {"field": lambda x: np.zeros(x.size)},
    {"field": lambda x: np.full(x.shape, np.inf)},
    {"field": lambda x: np.ones(x.shape, dtype=complex) * 1j},
])
def test_bad_inputs_fail_closed(change):
    args = dict(x=X, p3=P, pdg=13, h=1., field=ZERO)
    args.update(change)
    with pytest.raises(ValueError):
        ts.step(**args)


def test_all_rk_stages_field_results_validated():
    def bad_later(x):
        return np.zeros_like(x) if np.all(x[:, 0] == 0) else np.full_like(x, np.nan)
    with pytest.raises(ValueError, match="field result"):
        ts.step(X, P, 13, 1., bad_later)


def test_field_coverage_exception_propagates_unchanged():
    from beamfastsim.tsfield import CoverageError

    failure = CoverageError(np.array([False]))

    def uncovered(x):
        if np.any(x[:, 0] > 0.25):
            raise failure
        return np.zeros_like(x)

    with pytest.raises(CoverageError) as caught:
        ts.step(X, P, 13, 1., uncovered)
    assert caught.value is failure
    with pytest.raises(CoverageError) as caught:
        ts.plane_step(X, P, 13, 1., uncovered, [0.5, 0, 0], [1, 0, 0],
                      assume_monotone=True)
    assert caught.value is failure


@pytest.mark.parametrize("change", [
    {"assume_monotone": False}, {"h": 0}, {"x": X},
    {"p3": [[0, 100., 0]]}, {"normal": [2, 0, 0]},
    {"normal": [np.nan, 0, 0]}, {"origin": [0, 0]},
    {"direction": 2}, {"atol": 0}, {"atol": np.nan},
    {"maxiter": 0}, {"maxiter": 1.5},
])
def test_plane_rejects_unsupported_or_invalid(change):
    args = dict(x=[[-1., 0, 0]], p3=P, pdg=13, h=3., field=ZERO,
                origin=ZERO, normal=[1, 0, 0], assume_monotone=True)
    args.update(change)
    with pytest.raises(ValueError):
        ts.plane_step(**args)


def test_plane_start_within_tolerance_and_root_failure():
    with pytest.raises(ValueError, match="start on plane"):
        ts.plane_step([[-1e-12, 0, 0]], P, 13, 3., ZERO, ZERO, [1, 0, 0],
                      assume_monotone=True)
    with pytest.raises(RuntimeError, match="did not converge"):
        ts.plane_step([[-1., 0, 0]], P, 13, 3., ZERO, ZERO, [1, 0, 0],
                      assume_monotone=True, atol=1e-15, maxiter=1)


def test_empty_batch_and_zero_length_step():
    empty = ts.step(np.empty((0, 3)), np.empty((0, 3)), 13, 1., ZERO)
    assert empty.x.shape == (0, 3)
    assert empty.time.shape == (0,)
    zero = ts.step(X, P, 13, 0., ZERO, time=8.)
    np.testing.assert_array_equal(zero.x, X)
    np.testing.assert_array_equal(zero.p3, P)
    np.testing.assert_array_equal(zero.time, [8.])
    np.testing.assert_array_equal(zero.proper_time, [0.])


def test_import_is_lazy_and_step_never_imports_jax():
    # Run only when the parent executes tests: no subprocess is launched at
    # module collection. Block JAX imports even if installed on the test host.
    script = '''
import importlib.abc
import sys
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise AssertionError("unexpected JAX import")
sys.meta_path.insert(0, NoJax())
import beamfastsim.tsstep as ts
assert "mu2efastsim.ehelix" not in sys.modules
r = ts.step([[0, 0, 0]], [[100, 0, 0]], 13, 1, [0, 0, 1])
assert r.path[0] == 1
assert not any(n == "jax" or n.startswith("jax.") for n in sys.modules)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
    subprocess.run([sys.executable, "-c", script], env=env, check=True,
                   capture_output=True, text=True, timeout=30)


def test_plane_endpoint_roundoff_is_accepted_only_within_explicit_tolerance():
    result = ts.plane_step([[-1 - 5e-10, 0, 0]], P, 13, 1, ZERO, ZERO, [1, 0, 0],
                           assume_monotone=True, atol=1e-9)
    assert result.within_step[0]
    assert result.fraction[0] == 1
    assert abs(result.state.x[0, 0]) <= 1e-9
    strict = ts.plane_step([[-1 - 2e-9, 0, 0]], P, 13, 1, ZERO, ZERO, [1, 0, 0],
                           assume_monotone=True, atol=1e-9)
    assert not strict.within_step[0]


def test_representable_clock_with_extreme_mass_over_momentum():
    from beamfastsim.tsdecay import proper_time

    out = ts.step([[0., 0, 0]], [[1e-308, 0, 0]], 13, 1e-308, ZERO)
    expected = proper_time(1e-308, 1e-308, MMU)
    np.testing.assert_allclose(out.proper_time, expected, rtol=1e-14)
    np.testing.assert_allclose(out.time, expected, rtol=1e-14)
    assert np.all(np.isfinite(out.x))
