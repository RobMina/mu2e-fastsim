"""Unfinished-source summaries count every flagged lane and its kinetic energy."""
import numpy as np

from beamfastsim.flash_progress import describe


def test_remaining_energy_and_integer_identities(tmp_path):
    path = tmp_path / "births.npz"
    np.savez(path, source_pdg=np.array([11, -11, 11]), source_run=np.array([1, 1, 1]),
             source_subrun=np.array([2, 2, 3]), source_event=np.array([3, 4, 5]),
             capped=np.array([1, 0, 1], bool), loopers=np.array([0, 1, 0], bool),
             state_T=np.array([.04, 2., .08]), state_s=np.array([10., 30., 20.]))
    out = describe(path)
    assert out["capped"]["count"] == 2 and out["capped"]["parents"] == 3
    assert out["capped"]["remaining_kinetic_mev"] == .12
    assert out["loopers"]["remaining_kinetic_mev"] == 2.
    assert out["capped"]["rows"]["source_event"] == [3, 5]
