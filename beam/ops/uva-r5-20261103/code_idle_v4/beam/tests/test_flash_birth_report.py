"""Birth reports keep primary counter closure and geometric diagnostics separate."""
import numpy as np
import pytest
from beamfastsim import cechain as C
from beamfastsim import geometry as G
from beamfastsim.flash_birth_report import summarize


def _data():
    return dict(photon_valid=np.array([[True, False]]), photon_energy=np.array([[.5, 0]]),
                photon_position=np.array([[[1., 0, 5.], [0, 0, 0]]]),
                photon_direction=np.array([[[0., 0, 1], [0, 0, 0]]]), photon_mechanism=np.array([[1, 0]]),
                source_pdg=np.array([11]), count=np.array([1]), overflow=np.array([0]), capped=np.array([False]),
                loopers=np.array([False]), state_nph=np.array([1]), state_nan=np.array([0]),
                state_eph=np.array([.5]), state_ean=np.array([0.]))


def _geom():
    b = G.BeamGeometry(0., np.array([5.]), np.array([0.]), np.array([75.]), np.array([.05]),
                       -50., 1., 100., 200., 300., .5, 100., 200., 900., 100., 200.)
    return C.CEGeometry(b, 900., 900., 1.)


def test_complete_region_counts_and_energy():
    report = summarize(_data(), _geom())
    assert report["region_counts"] == {"foil": 1}
    assert report["region_energy_mev"] == {"foil": .5}
    assert report["retained_photons"] == report["emitted_photons"] == 1


@pytest.mark.parametrize("field,value", [("count", [2]), ("state_eph", [.6]), ("state_nph", [3])])
def test_counter_failures_reject(field, value):
    data = _data()
    data[field] = np.array(value)
    with pytest.raises(ValueError):
        summarize(data, _geom())
