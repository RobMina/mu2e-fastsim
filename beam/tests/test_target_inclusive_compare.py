"""Inclusive event joins retain multiplicity, pair daughters and transport-status failures."""
import numpy as np
import pytest

from beamfastsim import target_inclusive_compare as T


def ledger():
    return dict(identities=np.array([[1, 0, 2], [1, 0, 1]]), status=np.array([1, 0]),
                accepted=np.array([False, True]))


def test_exact_event_join_and_dio_multiplicity():
    rows = np.array([[1, 0, 1, 5], [1, 0, 2, 3], [1, 0, 1, 8]])
    index, summary = T.census(rows, {'pdg': np.array([11, 11, 11])}, ledger(), 166)
    assert index.tolist() == [1, 0, 1]
    assert summary['generated_electrons'] == 3
    assert summary['electron_multiplicity'] == {'1': 1, '2': 1}
    assert summary['status_counts'] == {'0': 1, '1': 1}
    assert not summary['all_status_zero']
    with pytest.raises(ValueError, match='absent'):
        T.event_indices(np.array([[1, 0, 3, 1]]), ledger()['identities'])
    with pytest.raises(ValueError, match='integer'):
        T.event_indices(np.array([[1, 0, 1.5, 1]]), ledger()['identities'])
    with pytest.raises(ValueError, match='duplicate'):
        T.event_indices(rows, np.array([[1, 0, 1], [1, 0, 1]]))


def test_rpc_pair_and_nonzero_status_contract():
    rows = np.array([[1, 0, 1, 5], [1, 0, 1, 6], [1, 0, 2, 5], [1, 0, 2, 6]])
    pdg = {'pdg': np.array([11, -11, 11, -11])}
    _, report = T.census(rows, pdg, ledger(), 179)
    assert report['generated_electrons'] == report['generated_positrons'] == 2
    with pytest.raises(ValueError, match='exactly one'):
        T.census(rows[:-1], {'pdg': pdg['pdg'][:-1]}, ledger(), 179)
    invalid = ledger()
    invalid['status'] = np.array([0, 10])
    with pytest.raises(ValueError, match='cannot pass'):
        T.census(rows, pdg, invalid, 179)


def test_streamed_inclusive_compare_keeps_rejected_identity(tmp_path, monkeypatch):
    uproot = pytest.importorskip('uproot')
    path, log = tmp_path / 'raw.root', tmp_path / 'ledger.npz'
    a = {k: np.ones(3, np.int32) for k in T.IDENTITY}
    a.update(subrun=np.zeros(3, np.int32), evt=np.array([1, 1, 2]), trk=np.array([1, 2, 1]),
             code=np.full(3, 166), pdg=np.full(3, 11), codestop=np.zeros(3))
    a.update({k: np.ones(3, np.float32) for k in T.START})
    for k in T.VD:
        a[k] = np.zeros((3, 13), np.float32)
    a['isvd'][:, 12] = 1
    a['pzvd'][:, 12] = 100
    a['xvd'][:, 12] = [460, 510, 560]
    with uproot.recreate(path) as root:
        root['cetree/ntpart'] = a
    np.savez(log, **ledger())
    with pytest.raises(ValueError, match='independent generation'):
        T.compare(path, log, None, lambda x: np.ones_like(x), 166)
    np.savez(log, generation_source_sha256=np.asarray('a' * 64), **ledger())
    monkeypatch.setattr(T.C, 'helix_rmax', lambda geom, x, *args: (x[:, 0], np.zeros(x.shape[0])))
    report, identity, columns = T.compare(path, log, None, lambda x: np.ones_like(x), 166)
    assert report['cuts']['450.0']['path_rejected_electrons'] == 1
    assert report['cuts']['450.0']['missed_identities'] == [[1, 0, 2, 1]]
    assert report['cuts']['450.0']['selected_nonzero_status_electrons'] == 1
    assert not report['filter_implication_certified']
    output = tmp_path / 'ref.npz'
    T.save_reference(output, identity, columns, 166, 2)
    with np.load(output) as r:
        assert r['rows'].tolist() == [3]
        assert r['gencount'].tolist() == [2]
        assert r['trk'].tolist() == [1, 2, 1]
