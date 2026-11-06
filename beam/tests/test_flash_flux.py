"""Synthetic exit-flux normalization and bookkeeping, not a G4 comparison."""
from types import SimpleNamespace

import numpy as np
import pytest

from beamfastsim.flash_flux import exit_records, read_states, summarize


def fixture():
    beam = SimpleNamespace(axis_x=0., foil_z=np.array([0., 4.]), foil_ht=np.array([.05, .1]),
                           foil_rin=np.array([1., 1.]), foil_rout=np.array([5., 6.]))
    payload = dict(source_run=np.ones(3, dtype=int), source_subrun=np.array([2, 2, 4]),
                   source_event=np.array([1, 1, 2]), source_particle_key=np.array([3, 4, 3]),
                   source_pdg=np.array([11, -11, 11]), source_time_ns=np.array([1e16, 1e16, 100.]),
                   photon_valid=np.array([[1, 0], [1, 1], [0, 0]], bool),
                   photon_position=np.array([[[2., 0., 0.], [0., 0., 0.]],
                                             [[2., 0., 4.], [0., 0., 8.]], [[0., 0., 0.]] * 2]),
                   photon_direction=np.tile([0., 0., 1.], (3, 2, 1)),
                   photon_energy=np.array([[1., 0.], [.1, 2.], [0., 0.]]),
                   photon_time=np.zeros((3, 2)), overflow=np.zeros(3, int), capped=np.zeros(3, bool),
                   loopers=np.array([False, False, True]))
    return payload, beam


def test_foils_and_nonfoil_births_and_late_times_are_kept():
    payload, beam = fixture()
    out = exit_records(payload, beam)
    np.testing.assert_array_equal(out["parent"], [0, 1, 1])
    np.testing.assert_array_equal(out["foil_index"], [0, 1, -1])
    np.testing.assert_allclose(out["path_in_foil_mm"][:2], [.05, .1])
    assert np.isnan(out["path_in_foil_mm"][-1])
    assert np.all(out["source_time_ns"] == 1e16)
    assert np.all((out["weight"][:2] > 0) & (out["weight"][:2] < 1))
    assert out["weight"][-1] == 0
    report, events = summarize(payload, out, source_events=3)
    assert report["source_events"] == 3 and report["represented_source_events"] == 2
    assert report["finite_foil_births"] == 2 and report["nonfoil_brems"] == 1
    assert report["loopers"] == 1 and report["parents"] == 3
    assert report["totals"]["transparent_photons"] == 2
    assert report["per_source_event"]["transparent_photons"] == pytest.approx(2 / 3)
    assert report["nonfoil_brems_energy_mev"] == 2.
    np.testing.assert_array_equal(events["transparent_photons"], [2., 0.])
    assert np.sum(report["histograms"]["uncollided"]) == pytest.approx(out["weight"].sum())


def test_brems_only_excludes_saved_annihilation():
    payload, beam = fixture()
    payload["photon_mechanism"] = np.array([[1, 0], [2, 1], [0, 0]])
    out = exit_records(payload, beam)
    assert len(out["energy"]) == 2
    report, _ = summarize(payload, out, source_events=2)
    assert report["all_recorded_photons"] == 3
    assert report["recorded_brems"] == 2 and report["finite_foil_births"] == 1


def test_unsupported_attenuation_is_reported_not_extrapolated():
    payload, beam = fixture()
    payload["photon_energy"][0, 0] = .00099
    out = exit_records(payload, beam)
    report, _ = summarize(payload, out, source_events=2)
    assert report["unsupported_attenuation_photons"] == 1
    assert report["unsupported_attenuation_energy_mev"] == .00099
    assert report["totals"]["transparent_photons"] == 2
    assert out["weight"][0] == 0


def test_zero_photon_events_remain_in_counts_and_denominator():
    payload, beam = fixture()
    payload["photon_valid"][:] = False
    out = exit_records(payload, beam)
    report, events = summarize(payload, out, source_events=4)
    assert report["source_events"] == 4 and report["parents"] == 3
    assert report["finite_foil_births"] == 0
    assert sum(report["histograms"]["uncollided"]) == 0
    np.testing.assert_array_equal(events["transparent_photons"], [0., 0.])


def test_bad_denominator_and_birth_payload():
    payload, beam = fixture()
    out = exit_records(payload, beam)
    for n in (0, 1, True, 2.1):
        with pytest.raises(ValueError, match="source_events"):
            summarize(payload, out, source_events=n)
    payload["photon_energy"][0, 0] = np.nan
    with pytest.raises(ValueError, match="payload"):
        exit_records(payload, beam)


def test_source_keeps_precision_and_identity(tmp_path):
    payload, _ = fixture()
    source = {k.removeprefix("source_"): v for k, v in payload.items() if k.startswith("source_")}
    source.update(x=np.zeros((3, 3)), p=np.tile([0., 0., 3.], (3, 1)))
    path = tmp_path / "states.npz"
    np.savez(path, **source)
    out = read_states(path)
    np.testing.assert_array_equal(out["time_ns"], source["time_ns"])
    source["event"] = source["event"].astype(float)
    np.savez(path, **source)
    with pytest.raises(ValueError, match="integer"):
        read_states(path)
