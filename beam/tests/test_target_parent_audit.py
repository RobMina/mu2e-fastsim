"""Contracts for parent multiplicities and sequential mixer exposure, including wraparound."""
import numpy as np
import pytest

from beamfastsim import target_parent_audit as A


def columns():
    return dict(run=np.ones(4, int), subrun=np.zeros(4, int), evt=np.array([7, 7, 8, 9]),
                trk=np.array([1, 2, 1, 1]), pdg=np.full(4, 13), codestop=np.full(4, 32),
                pstop=np.zeros(4), xstop=np.zeros(4), ystop=np.zeros(4), zstop=np.zeros(4), tstop=np.ones(4))


def test_parent_selection_matches_generator_and_preserves_order(monkeypatch):
    monkeypatch.setattr(A.G, "foil_index", lambda *a: np.zeros(4, int))
    a = columns()
    a["pstop"][0] = 1e-4
    out, r = A.reduce_parent(a, None, expected_events=3)
    assert out["event_multiplicity"].tolist() == [2, 1, 1]
    assert out["event_keys"][:, 2].tolist() == [7, 8, 9]
    assert r["eligible_muons"] == 4 and r["multiplicity"] == {"1": 2, "2": 1}
    assert r["expected_event_count_verified"] and not r["exact_production_electron_exposure_known"]
    assert r["max_end_momentum"] == 1e-4


def test_parent_refuses_incomplete_identity_or_target(monkeypatch):
    monkeypatch.setattr(A.G, "foil_index", lambda *a: np.full(4, -1))
    with pytest.raises(ValueError, match="outside staged"):
        A.reduce_parent(columns(), None, 3)
    with pytest.raises(ValueError, match="expected art entries"):
        A.reduce_parent(columns(), None, 4)
    a = columns()
    a["trk"][1] = 1
    with pytest.raises(ValueError, match="duplicate"):
        A.reduce_parent(a, None)
    a = columns()
    a["evt"] = np.array([7, 8, 7, 9])
    a["trk"] = np.array([1, 1, 2, 1])
    with pytest.raises(ValueError, match="contiguous"):
        A.reduce_parent(a, None)


def test_window_exposure_is_sequential_not_iid():
    r = A.window_exposure(np.array([2, 1, 1]), 2)
    assert r["min_electrons"] == 2 and r["max_electrons"] == 3
    assert r["mean_electrons"] == pytest.approx(8 / 3)
    assert r["window_sd"] == pytest.approx(np.std([3, 2, 3]))
    r = A.window_exposure(np.array([2, 1, 1]), 7)
    assert r["min_electrons"] == 9 and r["max_electrons"] == 10
    assert A.window_exposure(np.array([2, 1, 1]), 6)["window_sd"] == 0
    with pytest.raises(ValueError):
        A.window_exposure(np.array([0, 1]), 2)
    with pytest.raises(ValueError):
        A.window_exposure(np.array([1, 1]), 0)


def test_exact_birth_join_recovers_sequential_window_and_wraparound():
    parent = dict(xstop=np.array([10., 20., 21., 30.]), ystop=np.zeros(4), zstop=np.zeros(4),
                  event_index=np.array([0, 1, 1, 2]), event_multiplicity=np.array([1, 2, 1]))
    ref = dict(x=np.array([30., 10., 20., 21.]), y=np.zeros(4), z=np.zeros(4),
               evt=np.array([1, 2, 3, 3]), ifile=np.zeros(4, int))
    r = A.production_exposure(parent, ref, 3)
    assert r['files'][0]['inferred_first_parent_event_index'] == 2
    assert r['total_generated_electrons'] == 4
    assert r['exact_birth_matches'] == 4 and not r['generated_yield_validated']
    ref['x'][0] = 30.1
    with pytest.raises(ValueError, match='exactly one'):
        A.production_exposure(parent, ref, 3)
    ref['x'][0] = 20.
    with pytest.raises(ValueError, match='one sequential'):
        A.production_exposure(parent, ref, 3)


def test_fileindex_metadata_maps_sorted_extraction_to_physical_mixer_order(tmp_path):
    uproot = pytest.importorskip('uproot')
    ak = pytest.importorskip('awkward')
    path = tmp_path / 'parent.art'
    record = ak.Array({'eventID': {'subRun_': {'run_': {'run_': [1, 1, 1, 1]}, 'subRun_': [0, 0, 0, 0]},
                                  'event_': [0, 7, 8, 9]}, 'entry': [0, 2, 0, 1]})
    with uproot.recreate(path) as root:
        root['FileIndex'] = {'Element': record}
        root.mktree('Events', {'dummy': 'int32'}).extend({'dummy': np.zeros(3, np.int32)})
    parent = dict(event_keys=np.array([[1, 0, 7], [1, 0, 8], [1, 0, 9]]),
                  event_index=np.array([0, 0, 1, 2]), event_multiplicity=np.array([2, 1, 1]))
    out, r = A.physical_event_order(parent, path)
    assert out['event_index'].tolist() == [2, 2, 0, 1]
    assert out['event_keys'][:, 2].tolist() == [8, 9, 7]
    assert out['event_multiplicity'].tolist() == [1, 1, 2]
    assert r['reordered_events'] == 3
    swapped = tmp_path / 'byteswapped.art'
    raw = ak.to_numpy(record.entry).astype(np.int64).byteswap()
    event_id = dict(subRun_=dict(run_=dict(run_=np.array([1, 1, 1, 1], np.uint32).byteswap()),
                                subRun_=np.zeros(4, np.uint32)),
                    event_=np.array([np.iinfo(np.uint32).max, 7, 8, 9], np.uint32).byteswap())
    with uproot.recreate(swapped) as root:
        root['FileIndex'] = {'Element': ak.Array(dict(eventID=event_id, entry=raw))}
        root.mktree('Events', {'dummy': 'int32'}).extend({'dummy': np.zeros(3, np.int32)})
    with pytest.raises(ValueError):
        A.physical_event_order(parent, swapped)
    out, r = A.physical_event_order(parent, swapped, byteswap=True)
    assert out['event_multiplicity'].tolist() == [1, 1, 2]
    assert r['explicit_uproot_metadata_byteswap']
