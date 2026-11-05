"""Saved transport diagnostics preserve identities and expose, rather than repair, caps."""
import numpy as np
import pytest

from beamfastsim.flash_transport_report import summarize
from test_flash_birth_report import _data, _geom


def _saved():
    data = _data()
    data.update(source_run=np.array([1]), source_subrun=np.array([2]), source_event=np.array([3]),
                source_particle_key=np.array([4]), state_x=np.array([[1., 0., 4.95]]),
                state_u=np.array([[0., 0., 1.]]), state_T=np.array([1.]), state_s=np.array([5.]),
                state_t=np.array([.1]), state_cnt=np.zeros((1, 5), int), state_nvac=np.array([5]),
                state_nslab=np.array([0]), state_nstall=np.array([0]), state_nadapt=np.array([0]))
    return data


def test_capped_columns_preserve_source_identity_and_nearest_face():
    data = _saved()
    data["capped"][0] = True
    report = summarize(data, _geom())
    cols = report["capped_columns"]
    assert report["capped_parents"] == 1
    assert cols["parent_index"] == [0]
    assert cols["source_particle_key"] == [4]
    assert cols["nearest_foil_index"] == [0]
    np.testing.assert_allclose(cols["nearest_foil_face_distance_mm"], [0.])
    assert report["nonfoil_births"]["parent_index"] == []
    assert data["capped"][0]


def test_no_caps_and_nonfoil_birth_keeps_parent_index():
    data = _saved()
    data["photon_position"][0, 0, 2] = 10.
    report = summarize(data, _geom())
    assert report["capped_parent_quantiles"]["state_s"] is None
    assert report["nonfoil_births"]["region"] == ["other"]
    assert report["nonfoil_births"]["source_event"] == [3]


@pytest.mark.parametrize("field,value", [("state_x", [[np.nan, 0., 5.]]),
                                         ("state_u", [[0., 1.]]), ("count", [])])
def test_reject_invalid_saved_columns(field, value):
    data = _saved()
    data[field] = np.asarray(value)
    with pytest.raises(ValueError):
        summarize(data, _geom())


def test_loopers_preserve_full_identity_without_repairing_fate():
    data = _saved()
    data["loopers"][0] = True
    data["state_s"][0] = 30000.1
    report = summarize(data, _geom())
    assert report["loopers"] == 1
    assert report["capped_parents"] == 0
    assert report["looper_columns"]["source_particle_key"] == [4]
    assert report["looper_columns"]["state_s"] == [30000.1]
    assert data["loopers"][0]


def test_reference_comparison_checks_bytes_schema_and_values():
    from beamfastsim.flash_transport_report import compare
    data = _saved()
    same = {name: value.copy() for name, value in data.items()}
    assert compare(data, same)["exact"]
    same["state_T"][0] = 2.
    assert compare(data, same)["different_columns"] == ["state_T"]
    same.pop("state_T")
    assert compare(data, same)["extra_columns"] == ["state_T"]
    same["state_T"] = data["state_T"].astype(np.float32)
    assert compare(data, same)["different_columns"] == ["state_T"]
