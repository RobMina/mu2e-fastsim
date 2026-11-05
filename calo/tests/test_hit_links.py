"""Synthetic association audit including identical fits on different sensors."""
import numpy as np
import pytest

from test_controls import Branch, File, load_script


def audit_script(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "check_hit_weights", load_script("check_hit_weights"))
    return load_script("audit_hit_links")


class Tree:
    num_entries = 2

    def __init__(self, duplicate=False):
        self.branches = {
            "calohits": Branch([[{
                "nSiPMs_": 2, "crystalId_": 4, "time_": 12.,
                "timeErr_": np.float32(np.sqrt(.5)), "eDep_": 6.,
                "eDepErr_": np.float32(np.sqrt(2)),
                "recoDigis_": [0, 0] if duplicate else [0, 1],
            }], []]),
            "calorecodigis": Branch([[{
                "time_": 12., "timeErr_": 1., "eDep_": 6., "eDepErr_": 2.,
                "chi2_": 0., "ndf_": 5, "pileUp_": False, "SiPMID_": 8,
                "caloHitIdx_": 0,
            }, {
                "time_": 12., "timeErr_": 1., "eDep_": 6., "eDepErr_": 2.,
                "chi2_": 0., "ndf_": 5, "pileUp_": False, "SiPMID_": 9,
                "caloHitIdx_": -1 if duplicate else 0,
            }], []]),
        }

    def __getitem__(self, key):
        return self.branches[key]


def test_identical_fits_on_two_sensors_need_unique_identity(monkeypatch):
    script = audit_script(monkeypatch)
    monkeypatch.setattr(script.uproot, "open", lambda _: File(Tree(duplicate=True)))
    result = script.audit("unused", "EventNtuple/ntuple", 2)
    assert result["associations_clean"] is False
    assert result["repeated_hit_reco_pairs"] == 1
    assert result["reverse_multiplicity_bad_hits"] == 1
    assert result["unlinked_reco_with_duplicated_fit_signature"] == 1
    assert result["forward_reduction_diagnostic"]["time"]["value_float32_mismatches"] == 0
    assert result["forward_reduction_diagnostic"]["eDep"]["value_float32_mismatches"] == 0


def test_identical_fits_can_have_valid_distinct_links(monkeypatch):
    script = audit_script(monkeypatch)
    monkeypatch.setattr(script.uproot, "open", lambda _: File(Tree()))
    result = script.audit("unused", "EventNtuple/ntuple", 2)
    assert result["associations_clean"] is True
    assert result["repeated_hit_reco_pairs"] == 0
    assert result["unlinked_reco_digis"] == 0


def test_audit_refuses_out_of_event_forward_link(monkeypatch):
    import awkward as ak
    script = audit_script(monkeypatch)
    tree = Tree()
    hits = tree.branches["calohits"].data
    tree.branches["calohits"].data = ak.with_field(hits, ak.Array([[[0, 2]], []]), "recoDigis_")
    monkeypatch.setattr(script.uproot, "open", lambda _: File(tree))
    with pytest.raises(ValueError, match="outside its source event"):
        script.audit("unused", "EventNtuple/ntuple", 2)
