"""Production windows preserve exact stop exposure, including overlap and multiple wraps."""
import copy

import numpy as np
import pytest

from beamfastsim.target_window_stops import STOP_COLUMNS, select_windows


def fixture():
    parent = {k: np.zeros(4) for k in STOP_COLUMNS}
    parent.update(event_index=np.array([0, 1, 1, 2]), event_multiplicity=np.array([1, 2, 1]),
                  pdg=np.full(4, 13), codestop=np.full(4, 32), at_rest=np.ones(4, bool), foil=np.zeros(4, int))
    receipt = dict(production_exposure=dict(exact_production_electron_exposure_known=True,
        reference_sha256="bound-reference", total_generated_electrons=6,
        files=[dict(ifile=0, inferred_first_parent_event_index=2, generated_events=2, generated_electrons=2),
               dict(ifile=1, inferred_first_parent_event_index=0, generated_events=3, generated_electrons=4)]))
    return parent, receipt


def test_windows_preserve_wrapped_overlap_and_identity():
    parent, receipt = fixture()
    out, report = select_windows(parent, receipt)
    assert out["parent_stop_index"].tolist() == [0, 3, 0, 1, 2, 3]
    assert out["production_ifile"].tolist() == [0, 0, 1, 1, 1, 1]
    assert report["eligible_muons"] == 6 and not report["generated_yield_validated"]
    assert not report["filter_implication_certified"]
    receipt["production_exposure"]["files"] = [dict(ifile=0, inferred_first_parent_event_index=1,
        generated_events=7, generated_electrons=10)]
    receipt["production_exposure"]["total_generated_electrons"] = 10
    out, _ = select_windows(parent, receipt)
    assert np.bincount(out["parent_stop_index"]).tolist() == [2, 3, 3, 2]


def test_windows_refuse_inconsistent_receipt_or_parent():
    parent, receipt = fixture()
    bad = copy.deepcopy(receipt)
    bad["production_exposure"]["files"][0]["generated_electrons"] = 3
    with pytest.raises(ValueError, match="exact electron exposure"):
        select_windows(parent, bad)
    bad = copy.deepcopy(receipt)
    bad["production_exposure"]["reference_sha256"] = ""
    with pytest.raises(ValueError, match="reference hash"):
        select_windows(parent, bad)
    bad = copy.deepcopy(receipt)
    bad["production_exposure"]["exact_production_electron_exposure_known"] = False
    with pytest.raises(ValueError, match="exact production-window"):
        select_windows(parent, bad)
    parent["event_multiplicity"][0] = 2
    with pytest.raises(ValueError, match="multiplicities"):
        select_windows(parent, receipt)
