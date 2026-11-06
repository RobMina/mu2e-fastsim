"""Analytic survival checks and independent NIST total-attenuation anchors."""
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from beamfastsim import flash_attenuation as attenuation
from beamfastsim.flash_attenuation import al_mu_rho, uncollided_weight, write_xcom_csv


@pytest.mark.parametrize("energy,reference", [(0.1, 0.1704), (1., 0.06146), (20., 0.02168)])
def test_nist_public_table(energy, reference):
    # https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html
    # Four-significant-digit public table and separately rounded XCOM channels.
    assert al_mu_rho(energy) == pytest.approx(reference, rel=3e-4)


def test_every_bundled_table_node_and_range():
    table = np.loadtxt(Path(attenuation.__file__).parent / "data/flash_al_xcom.csv",
                       delimiter=",")
    assert table.shape == (82, 2)
    assert table[0, 0] == .001
    assert table[-1, 0] == 100000.
    assert np.all(np.diff(table[:, 0]) > 0)
    np.testing.assert_allclose(al_mu_rho(table[:, 0]), table[:, 1], rtol=3e-15)


def test_log_log_interpolation_and_array_shape():
    energy = np.array([[.1, np.sqrt(.1 * .15), .15]])
    mu = al_mu_rho(energy)
    assert mu.shape == (1, 3)
    assert mu[0, 1] == pytest.approx(np.sqrt(mu[0, 0] * mu[0, 2]), rel=2e-15)
    assert isinstance(al_mu_rho(1.), np.ndarray)
    assert al_mu_rho(1.).shape == ()
    assert al_mu_rho([]).shape == (0,)


def test_k_edge_is_not_smoothed_or_crossed():
    low, high = .0015595, .0015596
    mu = al_mu_rho([low, (low + high) / 2, high])
    factor = .60221367 / 26.981541
    assert mu[0] == pytest.approx((90.26 + 1.162 + 16130) * factor)
    assert np.isnan(mu[1])
    assert mu[2] == pytest.approx((90.26 + 1.162 + 177200) * factor)
    assert mu[2] > 10 * mu[0]
    assert np.isfinite(al_mu_rho(np.nextafter(low, 0)))
    assert np.isfinite(al_mu_rho(np.nextafter(high, 1)))
    assert np.isnan(al_mu_rho(np.nextafter(low, high)))
    assert np.isnan(al_mu_rho(np.nextafter(high, low)))


def test_no_clamp_or_extrapolation_including_current_brems_cut():
    bad = [np.nan, np.inf, -np.inf, -1., 0., .00099,
           np.nextafter(.001, 0), np.nextafter(100000., np.inf), .00155955]
    assert np.all(np.isnan(al_mu_rho(bad)))
    for distance in (0., 1.):
        result = uncollided_weight(bad, distance)
        assert not np.any(result["supported"])
        assert np.all(np.isnan(result["weight"]))


def test_analytic_mm_to_cm_and_default_density():
    # Independent 1 MeV sum: coherent + incoherent + photo (pairs are zero).
    mu = (.006443 + 2.746 + .001184) * .60221367 / 26.981541
    result = uncollided_weight(1., [1., 10.])
    np.testing.assert_allclose(result["weight"], np.exp(-mu * 2.699 * np.array([.1, 1.])),
                               rtol=2e-15)
    assert np.all(result["supported"])
    assert result["weight"][1] == pytest.approx(result["weight"][0] ** 10, rel=2e-15)


def test_zero_path_vacuum_broadcast_and_monotonicity():
    energies = np.array([[.1], [1.], [20.]])
    lengths = [0., 1., 10., 1e8]
    result = uncollided_weight(energies, lengths)
    assert result["weight"].shape == (3, 4)
    assert np.all(result["supported"])
    np.testing.assert_array_equal(result["weight"][:, 0], 1.)
    assert np.all(np.diff(result["weight"], axis=1) <= 0)
    np.testing.assert_array_equal(result["weight"][:, -1], 0.)
    np.testing.assert_array_equal(uncollided_weight(energies, lengths, 0.)["weight"], 1.)
    density = np.array([[1.], [2.]])
    result = uncollided_weight(1., [1., 10.], density)
    np.testing.assert_allclose(result["weight"],
                               np.exp(-al_mu_rho(1.) * density * np.array([.1, 1.])))
    assert uncollided_weight(.001, 0., 1e308)["weight"] == 1.


@pytest.mark.parametrize("bad", [-1., np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["distance_mm", "density_g_cm3"])
def test_invalid_lengths_and_densities_are_masked(bad, field):
    args = dict(energy=1., distance_mm=1., density_g_cm3=2.699)
    args[field] = [bad, 1.]
    result = uncollided_weight(**args)
    np.testing.assert_array_equal(result["supported"], [False, True])
    assert np.isnan(result["weight"][0])
    assert np.isfinite(result["weight"][1])


def _synthetic_xcom(tmp_path):
    energies = [1000., 1500., 1559.5, 1559.6, 1e11]
    channels = np.arange(1., 26.).reshape(5, 5)
    lines = ["13 26.981541", "1 5", "4", "K", "1559.6",
             " ".join(f"{e:13.5E}" for e in energies)]
    lines += [" ".join(f"{c:10.3E}" for c in channel) for channel in channels]
    lines += ["1", "3", "1.0E-3 1.5E-3 1.5596E-3", "53000 17930 16130"]
    source = tmp_path / "MDATX3.013"
    source.write_text("\n".join(lines) + "\n")
    return source, energies, channels


def test_parser_all_five_channels_units_and_deterministic_cli(tmp_path):
    source, energies, channels = _synthetic_xcom(tmp_path)
    out = tmp_path / "a.csv"
    write_xcom_csv(source, out)
    table = np.loadtxt(out, delimiter=",")
    np.testing.assert_allclose(table[:, 0], np.array(energies) / 1e6, rtol=0, atol=0)
    np.testing.assert_allclose(table[:, 1], channels.sum(axis=0) * .60221367 / 26.981541,
                               rtol=3e-15)
    second = tmp_path / "nested/b.csv"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    beam = str(Path(attenuation.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([beam, env.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, "-m", "beamfastsim.flash_attenuation",
                    "--xcom-data", str(source), "--out", str(second)],
                   check=True, env=env)
    assert out.read_bytes() == second.read_bytes()
    assert "NIST XCOM v3.1 MDATX3.013" in out.read_text()


@pytest.mark.parametrize("old,new", [("13 26.981541", "14 28.0855"),
                                       ("1 5", "2 5"), ("1 5", "1 500"),
                                       ("1.00000E+03", "9.00000E+02"),
                                       ("1.000E+00", "-1.00E+00")])
def test_parser_rejects_wrong_or_malformed_source(tmp_path, old, new):
    source, _, _ = _synthetic_xcom(tmp_path)
    source.write_text(source.read_text().replace(old, new, 1))
    with pytest.raises(ValueError):
        write_xcom_csv(source, tmp_path / "invalid.csv")
