"""RPC source tests independent of site files or transport."""
import json

import numpy as np
import pytest

from beamfastsim import rpcspec as S


def stops():
    return dict(run=np.array([1, 1, 1]), subrun=np.array([2, 2, 2]), evt=np.array([0, 1, 2]),
                pdg=np.array([-211, -211, 13]), at_rest=np.ones(3, bool), foil=np.array([0, 1, 2]),
                codestop=np.full(3, 78), xstop=np.array([1., 2., 3.]), ystop=np.array([4., 5., 6.]),
                zstop=np.array([6000., 6050., 6100.]), tstop=np.array([350., 420., 500.]))


def test_spectrum_bins_cut_and_chi2():
    spec = S.PhotonSpectrum()
    assert spec.edges.size == 896
    assert spec.edges[[0, -1]] == pytest.approx([50., 139.5])
    assert np.all(spec.pdf[spec.edges[:-1] > 134.53] == 0)
    sample = spec.sample(np.random.default_rng(42), 200_000)
    assert sample.min() >= 50 and sample.max() < 134.5
    counts = np.histogram(sample, spec.edges)[0]
    expected = len(sample) * spec.pdf / spec.pdf.sum()
    valid = expected >= 20
    chi2 = np.sum((counts[valid] - expected[valid]) ** 2 / expected[valid])
    assert chi2 / (valid.sum() - 1) < 1.3
    assert S.PhotonSpectrum(50, 50.25, 0.1).edges[-1] == pytest.approx(50.2)


@pytest.mark.parametrize("args", [(50, 40, .1), (50, 139.5, 0), (0, 139.5, .1),
                                  (50, np.nan, .1), (140, 145, .1), (50, 50.05, .1)])
def test_invalid_spectrum(args):
    with pytest.raises(ValueError):
        S.PhotonSpectrum(*args)


def test_pair_mass_shell_energy_correlations_and_isotropy():
    energy = np.linspace(50, 134, 8000)
    a = S.internal_pair(np.random.default_rng(4), energy)
    ee, ep = np.hypot(a["p"], S.ME), np.hypot(a["positron_p"], S.ME)
    assert np.allclose(ee + ep, energy, atol=1e-10)
    p3 = a["p"][:, None] * a["u"] + a["positron_p"][:, None] * a["positron_u"]
    assert np.allclose(energy ** 2 - np.sum(p3 ** 2, axis=1), a["pair_mass"] ** 2, atol=1e-8)
    eta = np.sqrt(1 - (2 * S.ME / a["pair_mass"]) ** 2)
    assert (np.abs(a["asymmetry"]) <= eta).all()
    assert np.allclose((ep - ee) / np.linalg.norm(p3, axis=1), a["asymmetry"])
    for key in ("u", "positron_u"):
        assert np.allclose(np.linalg.norm(a[key], axis=1), 1)
        assert np.abs(a[key].mean(axis=0)).max() < .025
        assert np.mean(a[key] ** 2, axis=0) == pytest.approx([1 / 3] * 3, abs=.015)


def test_pair_mass_histogram_against_integrated_kwj_density():
    from numpy.polynomial.legendre import leggauss
    energy = 100.
    edges = np.geomspace(2 * S.ME, energy, 17)
    z, w = leggauss(64)
    x = (edges[:-1, None] + edges[1:, None]) / 2 + np.diff(edges)[:, None] * z / 2
    eta = np.sqrt(1 - (2 * S.ME / x) ** 2)
    density = S.kwj_density(energy, x[:, :, None], eta[:, :, None] * z)
    integral_y = np.sum(density * w, axis=-1) * eta
    expected = np.sum(integral_y * w, axis=-1) * np.diff(edges) / 2
    n = 20_000
    expected *= n / expected.sum()
    a = S.internal_pair(np.random.default_rng(12), np.full(n, energy))
    observed = np.histogram(a["pair_mass"], edges)[0]
    assert np.sum((observed - expected) ** 2 / expected) / (len(expected) - 1) < 2.2


def test_kwj_support_and_envelope():
    energy = np.array([50., 100., 134.])[:, None, None]
    mass = np.linspace(2 * S.ME, 134., 300)[None, :, None]
    asymmetry = np.linspace(-1., 1., 81)[None, None, :]
    density = S.kwj_density(energy, mass, asymmetry)
    assert np.isfinite(density).all() and (density >= 0).all()
    assert (density <= S.kwj_density(energy, 2 * S.ME, 0) * (1 + 1e-12)).all()
    assert S.kwj_density(100., 101., 0.) == 0
    assert S.kwj_density(100., S.ME, 0.) == 0
    assert S.kwj_density(100., 2 * S.ME, 0.1) == 0


@pytest.mark.parametrize("energy", [[0.], [2 * S.ME], [np.nan], [np.inf], [[50.]], [S.M_AL + S.MPI]])
def test_invalid_pair_energy(energy):
    with pytest.raises(ValueError):
        S.internal_pair(np.random.default_rng(1), energy)


def test_stop_cuts_time_identity_and_repeatability():
    s = stops()
    a = S.from_stops(np.random.default_rng(3), s, 1000)
    b = S.from_stops(np.random.default_rng(3), s, 1000)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    idx = a["stop_idx"]
    assert set(idx) == {0, 1}
    assert np.array_equal(a["time"], s["tstop"][idx])
    assert np.array_equal(a["stop_evt"], s["evt"][idx])
    assert np.array_equal(a["x"], s["xstop"][idx])
    assert np.all(a["pdg"] == 11)
    assert np.allclose(a["positron_px"], a["positron_p"] * a["positron_u"][:, 0])
    assert np.all(S.from_stops(np.random.default_rng(3), s, 10, tmin=351)["stop_idx"] == 1)
    s["codestop"][1] = 77
    assert np.all(S.from_stops(np.random.default_rng(3), s, 10)["stop_idx"] == 0)
    s["at_rest"][0] = False
    with pytest.raises(ValueError, match="no selected"):
        S.from_stops(np.random.default_rng(3), s, 10)


def test_event_then_pion_sampling_does_not_weight_by_multiplicity():
    s = stops()
    take = np.r_[0, np.ones(10, int)]
    s = {k: v[take] for k, v in s.items()}
    a = S.from_stops(np.random.default_rng(2), s, 6000)
    assert np.mean(a["stop_evt"] == 0) == pytest.approx(.5, abs=.03)
    counts = np.bincount(a["stop_idx"], minlength=11)[1:]
    assert np.all(np.abs(counts - 300) < 70)


@pytest.mark.parametrize("change", ["muon", "weight", "missing", "shape", "nan", "float_event", "flag"])
def test_bad_stops(change):
    s = stops()
    if change == "muon":
        s["pdg"][:] = 13
    elif change == "weight":
        s["weight"] = np.ones(3)
    elif change == "missing":
        del s["evt"]
    elif change == "shape":
        s["xstop"] = s["xstop"][:1]
    elif change == "nan":
        s["tstop"][0] = np.nan
    elif change == "float_event":
        s["evt"] = np.array([.2, 1, 2])
    else:
        s["at_rest"] = np.ones(3, int)
    with pytest.raises(ValueError):
        S.from_stops(np.random.default_rng(1), s, 10)


def test_empty_generated_batch_has_no_lost_denominator_rows():
    a = S.from_stops(np.random.default_rng(1), stops(), 0)
    assert a["p"].size == 0 and a["u"].shape == (0, 3)


def test_cli(tmp_path):
    input_path, output_path = tmp_path / "stops.npz", tmp_path / "source.npz"
    np.savez(input_path, **stops())
    S.main(["--stops", str(input_path), "--n", "50", "--seed", "8", "--out", str(output_path)])
    with np.load(output_path, allow_pickle=False) as a:
        assert a["p"].size == 50 and a["positron_p"].size == 50
        meta = json.loads(str(a["metadata"]))
        assert meta["captures"] == 50 and meta["seed"] == 8
        assert meta["elow"] == 50 and meta["ehi"] == 139.5
