"""Limited RPC parent adapter fails closed rather than inventing a multiplicity normalization."""
import numpy as np
import pytest

from beamfastsim import target_rpc_parent_audit as R


def test_single_pion_parent_contract(monkeypatch):
    a = {k: np.zeros(3) for k in R.COLUMNS}
    a.update(run=np.ones(3), evt=np.array([8, 9, 10]), trk=np.ones(3), pdg=np.full(3, -211),
             codestop=np.full(3, 78), tstop=np.full(3, 350.))
    monkeypatch.setattr(R.G, "foil_index", lambda *a: np.zeros(3, int))
    out = R.reduce_parent(a, None, 3)
    assert out["event_multiplicity"].tolist() == [1, 1, 1]
    assert out["event_keys"][:, 2].tolist() == [8, 9, 10]
    a["pstop"][0] = 1e-4
    with pytest.raises(ValueError, match="nonzero"):
        R.reduce_parent(a, None, 3)
    a["pstop"][0] = 0
    a["tstop"][0] = 349.
    with pytest.raises(ValueError, match="exactly one"):
        R.reduce_parent(a, None, 3)
    with pytest.raises(ValueError, match="expected event"):
        R.reduce_parent(a, None, 4)


def test_reference_one_capture_one_electron_contract():
    ref = dict(creation_code=179, gencount=np.array([25000, 25000]), ifile=np.array([0, 1]),
               subrun=np.array([0, 1]), evt=np.array([1, 1]))
    assert R.validate_reference(ref, 2).tolist() == [25000, 25000]
    ref["creation_code"] = 166
    with pytest.raises(ValueError, match="RPC internal"):
        R.validate_reference(ref, 2)
    ref["creation_code"] = 179
    ref.update(ifile=np.array([0, 0, 1]), subrun=np.array([0, 0, 1]), evt=np.array([1, 1, 1]))
    with pytest.raises(ValueError, match="one primary electron"):
        R.validate_reference(ref, 2)


def test_repeated_stops_resolve_only_by_exact_anchored_mixer_phase():
    parent = dict(xstop=np.array([10., 20., 10.]), ystop=np.zeros(3), zstop=np.zeros(3),
                  tstop=np.array([350., 360., 350.]), event_index=np.arange(3), event_multiplicity=np.ones(3, int))
    ref = dict(x=np.array([20., 10., 10., 20.]), y=np.zeros(4), z=np.zeros(4), time=np.array([360., 350., 350., 360.]),
               evt=np.arange(1, 5), ifile=np.zeros(4, int))
    r = R.exact_window(parent, ref, 4)
    assert r['inferred_first_parent_event_index'] == 1
    assert r['matched_electrons'] == 4 and r['unique_phase_anchor_rows'] == 2
    assert r['repeated_stop_rows_resolved_by_exact_phase'] == 2
    ref['time'][1] = 351.
    with pytest.raises(ValueError, match='birth times'):
        R.exact_window(parent, ref, 4)
    ref['time'][1] = 350.
    ref['x'][1] = 20.
    with pytest.raises(ValueError, match='sequential'):
        R.exact_window(parent, ref, 4)


def test_ambiguous_birth_event_id_cannot_be_silently_truncated():
    parent = dict(xstop=np.array([10., 20., 10.]), ystop=np.zeros(3), zstop=np.zeros(3),
                  tstop=np.array([350., 360., 350.]), event_index=np.arange(3), event_multiplicity=np.ones(3, int))
    ref = dict(x=np.array([20., 10.]), y=np.zeros(2), z=np.zeros(2), time=np.array([360., 350.]),
               evt=np.array([1., 2.5]), ifile=np.zeros(2, int))
    with pytest.raises(ValueError, match='integer generated'):
        R.exact_window(parent, ref, 4)
